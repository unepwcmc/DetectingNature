#include "ClassificationFramework.h"
using namespace std;

ClassificationFramework::ClassificationFramework(string datasetPath,
		Settings &settings, bool skipCache) {
		
	m_skipCache = skipCache;
	m_settings = settings;
	
	m_cacheHelper = new CacheHelper(datasetPath, m_settings);
	
	m_datasetManager = m_skipCache ?
		nullptr : m_cacheHelper->load<DatasetManager>("dataset");
	if(m_datasetManager == nullptr) {
		m_datasetManager = new DatasetManager(datasetPath,
			settings.trainImagesPerClass);
		m_cacheHelper->save<DatasetManager>("dataset", m_datasetManager);
	}
	
	m_featureExtractor = new FeatureExtractor(m_settings.featureType,
		m_settings.gridSpacing, m_settings.patchSize);
}

ClassificationFramework::~ClassificationFramework() {
	delete m_datasetManager;
	delete m_featureExtractor;
}

ImageFeatures* ClassificationFramework::extractFeature(string imagePath) {
	ImageFeatures* features = m_cacheHelper->load<ImageFeatures>(imagePath);
	if(features == nullptr) {
		Image img(imagePath, m_settings.colourspace);
		features = m_featureExtractor->extract(img);
		m_cacheHelper->save<ImageFeatures>(imagePath, features);
	}

	return features;
}

Codebook* ClassificationFramework::prepareCodebook(
		vector<string> imagePaths, bool skipCache) {
		
	vector<ImageFeatures*> features;
	features.resize(m_settings.textonImages, nullptr);

	#pragma omp parallel for
	for(unsigned int i = 0; i < m_settings.textonImages; i++) {
		features[i] = extractFeature(imagePaths[i]);
	}

	Codebook* codebook = skipCache ?
		nullptr : m_cacheHelper->load<Codebook>("codebook");
	if(codebook == nullptr) {
		CodebookGenerator codebookGenerator(features);
		codebook = codebookGenerator.generate(m_settings.textonImages,
			m_settings.codewords, m_settings.histogramType);
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
			m_settings.pyramidLevels);
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
		vector<Histogram*> trainHistograms) {
		
	vector<string> classNames = m_datasetManager->listClasses();
	Classifier* classifier = new Classifier(classNames);
	classifier->train(trainHistograms,
		m_datasetManager->getTrainClasses(), m_settings.C);
	
	return classifier;
}

double ClassificationFramework::run() {
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
