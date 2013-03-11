#include "ClassificationFramework.h"
using namespace std;

ClassificationFramework::ClassificationFramework(Settings &settings) {
	m_settings = settings;
	
	m_cacheHelper = new CacheHelper(m_settings);
	
	m_datasetManager = m_cacheHelper->load<DatasetManager>("dataset");
	if(m_datasetManager == nullptr) {
		m_datasetManager = new DatasetManager(settings.datasetPath,
			settings.trainImagesPerClass);
		m_cacheHelper->save<DatasetManager>("dataset", m_datasetManager);
	}
}

ClassificationFramework::~ClassificationFramework() {
	delete m_datasetManager;
}

vector<ImageFeatures*> ClassificationFramework::generateFeatures(
		vector<string> imagePaths) {
		
	OutputHelper::printMessage("Extracting features:");
	
	FeatureExtractor featureExtractor(m_settings.featureType,
		m_settings.gridSpacing, m_settings.patchSize);
	vector<ImageFeatures*> features(imagePaths.size(), nullptr);

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < imagePaths.size(); i++) {
		features[i] = m_cacheHelper->load<ImageFeatures>(imagePaths[i]);
		if(features[i] == nullptr) {
			Image img(imagePaths[i], m_settings.colourspace);
			features[i] = featureExtractor.extract(img);
			m_cacheHelper->save<ImageFeatures>(imagePaths[i], features[i]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Processing image "
				+ DatasetManager::getFilename(imagePaths[i]),
				currentIter, imagePaths.size());
		}
	}

	return features;
}

vector<Histogram*> ClassificationFramework::generateHistograms(
		vector<string> imagePaths) {
		
	vector<Histogram*> histograms(imagePaths.size(), nullptr);
	vector<ImageFeatures*> features = generateFeatures(imagePaths);
	
	OutputHelper::printMessage("Generating histograms:");
			
	Codebook* codebook = m_cacheHelper->load<Codebook>("codebook");
	if(codebook == nullptr) {
		CodebookGenerator codebookGenerator(features);
		codebook = codebookGenerator.generate(
			m_settings.textonImages, m_settings.codewords);
		m_cacheHelper->save<Codebook>("codebook", codebook);
	}

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < imagePaths.size(); i++) {	
		histograms[i] = m_cacheHelper->load<Histogram>(imagePaths[i]);
		if(histograms[i] == nullptr) {
			histograms[i] = codebook->computeHistogram(features[i],
				m_settings.pyramidLevels);
			m_cacheHelper->save<Histogram>(imagePaths[i], histograms[i]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Processing image "
				+ DatasetManager::getFilename(imagePaths[i]),
				currentIter, imagePaths.size());
		}
	}
	
	delete codebook;
	for(unsigned int i = 0; i < histograms.size(); i++)
		delete features[i];
			
	return histograms;
}

double ClassificationFramework::trainClassifier() {
	vector<Histogram*> trainHistograms =
		generateHistograms(m_datasetManager->getTrainData());

	vector<string> classNames = m_datasetManager->listClasses();
	Classifier classifier(classNames);
	classifier.train(trainHistograms,
		m_datasetManager->getTrainClasses(), m_settings.C);
	
	vector<Histogram*> testHistograms =
		generateHistograms(m_datasetManager->getTestData());
	double result =
		classifier.test(testHistograms, m_datasetManager->getTestClasses());
	
	for(unsigned int i = 0; i < trainHistograms.size(); i++)
		delete trainHistograms[i];
	for(unsigned int i = 0; i < testHistograms.size(); i++)
		delete testHistograms[i];

	return result;
}

double ClassificationFramework::run() {	
	return trainClassifier();
}
