#include "ClassificationFramework.h"
using namespace std;
using namespace boost::filesystem;

BOOST_CLASS_EXPORT(FisherCodebook);
BOOST_CLASS_EXPORT(KMeansCodebook);

typedef boost::function<FeatureExtractor*(const SettingsManager*)>
	featureFactory_t;
typedef boost::function<FeatureTransform*(const SettingsManager*)>
	transformFactory_t;
typedef boost::function<ImageLoader*(const SettingsManager*)>
	loaderFactory_t;
typedef boost::function<CodebookGenerator*(const SettingsManager*)>
	codebookFactory_t;
typedef boost::function<Classifier*(const SettingsManager*,
	std::vector<std::string>)> classifierFactory_t;

ClassificationFramework::ClassificationFramework(string datasetPath,
		const SettingsManager* settings, bool skipCache) {
		
	m_skipCache = skipCache;
	m_settings = settings;
	
	m_cacheHelper = new CacheHelper(datasetPath, m_settings);
	
	m_datasetManager = m_skipCache ?
		nullptr : m_cacheHelper->load<DatasetManager>("dataset");
	if(m_datasetManager == nullptr) {
		m_datasetManager = new DatasetManager(datasetPath,
			m_settings->get<int>("classifier.trainImagesPerClass"));
		m_cacheHelper->save<DatasetManager>("dataset", m_datasetManager);
	}
	
	// Initialize factory maps
	
	map<string, loaderFactory_t> loaderFactories;
	loaderFactories["Greyscale"] = boost::factory<GreyscaleImageLoader*>();
	loaderFactories["Opponent"] = boost::factory<OpponentImageLoader*>();
	loaderFactories["HSV"] = boost::factory<HSVImageLoader*>();
	
	map<string, featureFactory_t> featureFactories;
	featureFactories["SIFT"] = boost::factory<SIFTFeatureExtractor*>();
	featureFactories["HOG"] = boost::factory<HOGFeatureExtractor*>();
	featureFactories["LBP"] = boost::factory<LBPFeatureExtractor*>();
	
	map<string, transformFactory_t> transformFactories;
	transformFactories["Hellinger"] =
		boost::factory<HellingerFeatureTransform*>();
	
	map<string, codebookFactory_t> codebookFactories;
	codebookFactories["Fisher"] =
		boost::factory<FisherCodebookGenerator*>();
	codebookFactories["KMeans"] =
		boost::factory<KMeansCodebookGenerator*>();
	
	map<string, classifierFactory_t> classifierFactories;
	classifierFactories["Linear"] = boost::factory<LinearClassifier*>();
	classifierFactories["SVM"] = boost::factory<SVMClassifier*>();

	// Create instances from the settings file using the factory maps
	m_imageLoader =
		loaderFactories[m_settings->get<string>("image.type")](m_settings);
	
	m_featureExtractor =
		featureFactories[m_settings->get<string>("features.type")](m_settings);
	
	vector<string> transformList =
		m_settings->get<vector<string> >("features.transforms");
	for(unsigned int i = 0; i < transformList.size(); i++) {
		m_featureTransforms.push_back(
			transformFactories[transformList[i]](m_settings));
	}
	
	m_codebookGenerator =
		codebookFactories[m_settings->get<string>("codebook.type")](m_settings);
	
	vector<string> classNames = m_datasetManager->listClasses();
	m_classifier = classifierFactories[
		m_settings->get<string>("classifier.type")](m_settings, classNames);
}

ClassificationFramework::~ClassificationFramework() {
	delete m_datasetManager;
	delete m_featureExtractor;
	delete m_classifier;
	
	for(unsigned int i = 0; i < m_featureTransforms.size(); i++) {
		delete m_featureTransforms[i];
	}
}

ImageFeatures* ClassificationFramework::extractFeature(string imagePath) {
	ImageFeatures* features = m_cacheHelper->load<ImageFeatures>(imagePath);
	if(features == nullptr) {
		// Extract features
		ImageData* img = m_imageLoader->loadImage(imagePath);
		features = m_featureExtractor->extract(img);
		delete img;
		
		// Apply transformations
		for(unsigned int i = 0; i < m_featureTransforms.size(); i++) {
			features = m_featureTransforms[i]->transform(features);
		}
		m_cacheHelper->save<ImageFeatures>(imagePath, features);
	}

	return features;
}

Codebook* ClassificationFramework::prepareCodebook(
		vector<string> imagePaths, bool skipCache) {

	Codebook* codebook = skipCache ?
		nullptr : m_cacheHelper->load<Codebook>("codebook");
	if(codebook == nullptr) {
		unsigned int numTextonImages =
			m_settings->get<unsigned int>("codebook.textonImages");
		numTextonImages = min(imagePaths.size(), numTextonImages);
		vector<ImageFeatures*> features;
		features.resize(numTextonImages, nullptr);
	
		unsigned int currentIter = 0;
		#pragma omp parallel for
		for(unsigned int i = 0; i < numTextonImages; i++) {
			features[i] = extractFeature(imagePaths[i]);
		
			#pragma omp critical
			{
				currentIter++;
				OutputHelper::printProgress("Processing image "
					+ DatasetManager::getFilename(imagePaths[i]),
					currentIter, numTextonImages);
			}
		}
		
		codebook = m_codebookGenerator->generate(features);
		m_cacheHelper->save<Codebook>("codebook", codebook);
		
		for(unsigned int i = 0; i < features.size(); i++) {
			delete features[i];
		}
	}
	
	return codebook;
}

Histogram* ClassificationFramework::generateHistogram(
		Codebook* codebook, string imagePath) {

	Histogram* histogram = m_skipCache ?
		nullptr : m_cacheHelper->load<Histogram>(imagePath);
	if(histogram == nullptr) {
		ImageFeatures* features = extractFeature(imagePath);
		histogram = codebook->encode(features);
		m_cacheHelper->save<Histogram>(imagePath, histogram);
		delete features;
	}

	return histogram;
}

vector<Histogram*> ClassificationFramework::generateHistograms(
		vector<string> imagePaths, bool skipCodebook) {
		
	OutputHelper::printMessage("Generating histograms:");
		
	Codebook* codebook = prepareCodebook(imagePaths, skipCodebook);
	vector<Histogram*> histograms(imagePaths.size(), nullptr);

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < imagePaths.size(); i++) {	
		histograms[i] = generateHistogram(codebook, imagePaths[i]);
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Processing image "
				+ DatasetManager::getFilename(imagePaths[i]),
				currentIter, imagePaths.size());
		}
	}
	
	delete codebook;			
	return histograms;
}

void ClassificationFramework::train() {
	for(unsigned int i = 0; i < m_trainHistograms.size(); i++)
		delete m_trainHistograms[i];
	
	m_trainHistograms =
		generateHistograms(m_datasetManager->getTrainData(), m_skipCache);

	m_classifier->train(m_trainHistograms, m_datasetManager->getTrainClasses());
}

double ClassificationFramework::testRun() {
	vector<string> classNames = m_datasetManager->listClasses();
	vector<string> imagePaths = m_datasetManager->getTestData();
	vector<unsigned int> testClasses = m_datasetManager->getTestClasses();
	Codebook* codebook = prepareCodebook(imagePaths, false);
	
	OutputHelper::printMessage("Testing Classifier:");
	ConfusionMatrix confMat(classNames);
	
	double avgTime = 0.0;
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < imagePaths.size(); i++) {
		clock_t start = clock();
		Histogram* testHist = generateHistogram(codebook, imagePaths[i]);
		pair<unsigned int, double> result = m_classifier->classify(testHist);
		delete testHist;
		avgTime += (clock() - start) / (double) CLOCKS_PER_SEC;
		
		confMat.addEntry(testClasses[i], result.first);
			
		#pragma omp critical
		{
			currentIter++;
						
			if(!(currentIter % 100)) {
				ConfusionMatrix tempMat = confMat;
				cout << endl;
				tempMat.printMatrix();
				cout << "    Taking " <<
					((avgTime / 100.0) * 1000.0) << "ms per image" << endl;
				avgTime = 0.0;
			}

			OutputHelper::printResults("Predicting image", currentIter,
				imagePaths.size(), result.first, result.second);
		}
	}
	confMat.printMatrix();

	return confMat.getDiagonalAverage();
}

vector<ClassificationFramework::Result> ClassificationFramework::classify(
		string imagesFolder) {
			
	vector<string> imagePaths;
	if(is_directory(imagesFolder)) {
		for(directory_iterator it(imagesFolder);
				it != directory_iterator(); it++) {
				
			imagePaths.push_back(it->path().relative_path().string());
		}
	} else {
		imagePaths.push_back(imagesFolder);
	}
	
	Codebook* codebook = prepareCodebook(imagePaths, false);
	
	vector<string> classNames = m_datasetManager->listClasses();
	vector<Result> results;
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < imagePaths.size(); i++) {
		Histogram* testHist = generateHistogram(codebook, imagePaths[i]);
		pair<unsigned int, double> resultClass
			= m_classifier->classify(testHist);
		Result result;
		result.filepath = imagePaths[i];
		result.category = classNames[resultClass.first];
		result.certainty = resultClass.second;
		delete testHist;
		
		#pragma omp critical
		{
			results.push_back(result);
			currentIter++;
			OutputHelper::printResults("Classifying image", currentIter,
				imagePaths.size(), resultClass.first, resultClass.second);
		}
	}

	return results;
}
