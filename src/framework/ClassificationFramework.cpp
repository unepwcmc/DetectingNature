#include "ClassificationFramework.h"
using namespace std;
using namespace boost::filesystem;

typedef boost::function<FeatureExtractor*(const SettingsManager*)>
	featureFactory_t;
typedef boost::function<FeatureTransform*(const SettingsManager*)>
	transformFactory_t;
typedef boost::function<ImageLoader*(const SettingsManager*)>
	loaderFactory_t;
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
	
	map<string, classifierFactory_t> classifierFactories;
	classifierFactories["Linear"] = boost::factory<LinearClassifier*>();
	classifierFactories["SVM"] = boost::factory<SVMClassifier*>();

	// Create instances from the settings file using the factory maps
	m_imageLoader =
		loaderFactories[m_settings->get<string>("image.type")](m_settings);
	
	m_featureExtractor =
		featureFactories[m_settings->get<string>("features.type")](m_settings);
	
	vector<string> transformList;
	string transforms = m_settings->get<string>("features.transforms");
	boost::split(transformList, transforms, boost::is_any_of(" |"));
	for(unsigned int i = 0; i < transformList.size(); i++) {
		if(transformList[i].length() > 0) {
			m_featureTransforms.push_back(
				transformFactories[transformList[i]](m_settings));
		}
	}
	
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
		
	vector<ImageFeatures*> features;
	features.resize(m_settings->get<int>("codebook.textonImages"), nullptr);

	Codebook* codebook = skipCache ?
		nullptr : m_cacheHelper->load<Codebook>("codebook");
	if(codebook == nullptr) {
		unsigned int currentIter = 0;
		#pragma omp parallel for
		for(unsigned int i = 0;
				i < m_settings->get<unsigned int>("codebook.textonImages"); i++) {
				
			features[i] = extractFeature(imagePaths[i]);
		
			#pragma omp critical
			{
				currentIter++;
				OutputHelper::printProgress("Processing image "
					+ DatasetManager::getFilename(imagePaths[i]),
					currentIter, m_settings->get<int>("codebook.textonImages"));
			}
		}
		
		CodebookGenerator codebookGenerator(features);
		codebook = codebookGenerator.generate(
			m_settings->get<int>("codebook.textonImages"),
			m_settings->get<int>("codebook.codewords"),
			(Codebook::Type)0);
			//m_settings->get<string>("histogram.type"));
		m_cacheHelper->save<Codebook>("codebook", codebook);
	}
	
	for(unsigned int i = 0; i < features.size(); i++) {
		delete features[i];
	}
	
	return codebook;
}

Histogram* ClassificationFramework::generateHistogram(
		Codebook* codebook, string imagePath) {

	Histogram* histogram = m_skipCache ?
		nullptr : m_cacheHelper->load<Histogram>(imagePath);
	if(histogram == nullptr) {
		ImageFeatures* features = extractFeature(imagePath);
		histogram = codebook->computeHistogram(features,
			m_settings->get<int>("histogram.pyramidLevels"));
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

double ClassificationFramework::testRun() {
	vector<Histogram*> trainHistograms =
		generateHistograms(m_datasetManager->getTrainData(), m_skipCache);

	m_classifier->train(trainHistograms, m_datasetManager->getTrainClasses());
	
	vector<Histogram*> testHistograms =
		generateHistograms(m_datasetManager->getTestData(), false);
	double result =
		m_classifier->test(testHistograms, m_datasetManager->getTestClasses());
	
	for(unsigned int i = 0; i < trainHistograms.size(); i++)
		delete trainHistograms[i];
	for(unsigned int i = 0; i < testHistograms.size(); i++)
		delete testHistograms[i];

	return result;
}

map<string, string> ClassificationFramework::classify(string imagesFolder) {
	vector<Histogram*> trainHistograms =
		generateHistograms(m_datasetManager->getTrainData(), m_skipCache);

	m_classifier->train(trainHistograms, m_datasetManager->getTrainClasses());
	
	vector<string> filePaths;
	for(directory_iterator it(imagesFolder); it != directory_iterator(); it++) {
		filePaths.push_back(it->path().relative_path().string());
	}
	
	vector<Histogram*> testHistograms = generateHistograms(filePaths, false);
	
	vector<string> classNames = m_datasetManager->listClasses();
	map<string, string> results;
	#pragma omp parallel for
	for(unsigned int i = 0; i < testHistograms.size(); i++) {
		pair<unsigned int, double> resultClass
			= m_classifier->classify(testHistograms[i]);
		results[filePaths[i]] = classNames[resultClass.first];
	}
	
	for(unsigned int i = 0; i < trainHistograms.size(); i++)
		delete trainHistograms[i];
	for(unsigned int i = 0; i < testHistograms.size(); i++)
		delete testHistograms[i];

	return results;
}
