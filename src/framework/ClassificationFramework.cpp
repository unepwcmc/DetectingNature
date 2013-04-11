#include "ClassificationFramework.h"
using namespace std;
using namespace boost::filesystem;

typedef boost::function<FeatureExtractor*(const SettingsManager*)>
	featureFactory_t;
typedef boost::function<FeatureTransform*(const SettingsManager*)>
	transformFactory_t;

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
	map<string, featureFactory_t> featureFactories;
	featureFactories["SIFT"] = boost::factory<SIFTFeatureExtractor*>();
	featureFactories["HOG"] = boost::factory<HOGFeatureExtractor*>();
	featureFactories["LBP"] = boost::factory<LBPFeatureExtractor*>();
	
	map<string, transformFactory_t> transformFactories;
	transformFactories["Hellinger"] =
		boost::factory<HellingerFeatureTransform*>();

	// Create instances from the settings file using the factory maps
	m_featureExtractor =
		featureFactories[m_settings->get<string>("features.type")](m_settings);
	
	vector<string> transformList;
	string transforms = m_settings->get<string>("features.transforms");
	boost::split(transformList, transforms, boost::is_any_of(" |"));
	for(unsigned int i = 0; i < transformList.size(); i++) {
		m_featureTransforms.push_back(
			transformFactories[transformList[i]](m_settings));
	}
}

ClassificationFramework::~ClassificationFramework() {
	delete m_datasetManager;
	delete m_featureExtractor;
}

ImageFeatures* ClassificationFramework::extractFeature(string imagePath) {
	ImageFeatures* features = m_cacheHelper->load<ImageFeatures>(imagePath);
	if(features == nullptr) {
		// Extract features
		Image img(imagePath,
			(Image::Colourspace)m_settings->get<int>("image.colourspace"));
		features = m_featureExtractor->extract(img);
		
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
			(Codebook::Type)m_settings->get<int>("histogram.type"));
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

Classifier* ClassificationFramework::trainClassifier(
		std::vector<Histogram*> trainHistograms) {
		
	vector<string> classNames = m_datasetManager->listClasses();
	Classifier* classifier = new Classifier(classNames);
	classifier->train(trainHistograms,
		m_datasetManager->getTrainClasses(),
		m_settings->get<float>("classifier.c"));
	
	return classifier;
}

double ClassificationFramework::testRun() {
	vector<Histogram*> trainHistograms =
		generateHistograms(m_datasetManager->getTrainData(), m_skipCache);

	Classifier* classifier = trainClassifier(trainHistograms);
	
	vector<Histogram*> testHistograms =
		generateHistograms(m_datasetManager->getTestData(), false);
	double result =
		classifier->test(testHistograms, m_datasetManager->getTestClasses());
	
	for(unsigned int i = 0; i < trainHistograms.size(); i++)
		delete trainHistograms[i];
	for(unsigned int i = 0; i < testHistograms.size(); i++)
		delete testHistograms[i];

	return result;
}

map<string, string> ClassificationFramework::classify(string imagesFolder) {
	vector<Histogram*> trainHistograms =
		generateHistograms(m_datasetManager->getTrainData(), m_skipCache);

	Classifier* classifier = trainClassifier(trainHistograms);
	
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
			= classifier->classify(testHistograms[i]);
		results[filePaths[i]] = classNames[resultClass.first];
	}
	
	for(unsigned int i = 0; i < trainHistograms.size(); i++)
		delete trainHistograms[i];
	for(unsigned int i = 0; i < testHistograms.size(); i++)
		delete testHistograms[i];

	return results;
}
