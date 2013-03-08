#include "ClassificationFramework.h"
using namespace std;

ClassificationFramework::ClassificationFramework(Settings &settings) {
	m_settings = settings;
	
	m_cacheHelper = new CacheHelper(m_settings);

	m_cachePath = "cache/" + settings.datasetPath;
	if(!boost::filesystem::exists(m_cachePath)) {
		boost::filesystem::create_directories(m_cachePath);
	}
	
	m_datasetManager = m_cacheHelper->load<DatasetManager>("dataset");
	if(m_datasetManager == nullptr) {
		m_datasetManager = new DatasetManager(settings.datasetPath);
		m_cacheHelper->save<DatasetManager>("dataset", m_datasetManager);
	}
	
	m_imagePaths = m_datasetManager->listFiles();
}

ClassificationFramework::~ClassificationFramework() {
	delete m_datasetManager;
}

vector<ImageFeatures*> ClassificationFramework::generateFeatures() {
	OutputHelper::printMessage("Extracting features:");
	
	FeatureExtractor featureExtractor(m_settings.featureType,
		m_settings.gridSpacing, m_settings.patchSize);
	vector<ImageFeatures*> features(m_imagePaths.size(), nullptr);	

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_imagePaths.size(); i++) {
		features[i] = m_cacheHelper->load<ImageFeatures>(m_imagePaths[i]);
		if(features[i] == nullptr) {
			Image img(m_imagePaths[i], m_settings.colourspace);
			features[i] = featureExtractor.extract(img);
			m_cacheHelper->save<ImageFeatures>(m_imagePaths[i], features[i]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Processing image "
				+ DatasetManager::getFilename(m_imagePaths[i]),
				currentIter, m_imagePaths.size());
		}
	}

	return features;
}

vector<Histogram*> ClassificationFramework::generateHistograms() {
	vector<Histogram*> histograms(m_imagePaths.size(), nullptr);
	vector<ImageFeatures*> features = generateFeatures(); 
	
	OutputHelper::printMessage("Generating histograms:");
			
	CodebookGenerator codebookGenerator(features);
	Codebook* codebook = codebookGenerator.generate(
		m_settings.textonImages, m_settings.codewords);

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_imagePaths.size(); i++) {	
		histograms[i] = m_cacheHelper->load<Histogram>(m_imagePaths[i]);
		if(histograms[i] == nullptr) {
			histograms[i] = codebook->computeHistogram(features[i],
				m_settings.pyramidLevels);
			m_cacheHelper->save<Histogram>(m_imagePaths[i], histograms[i]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Processing image "
				+ DatasetManager::getFilename(m_imagePaths[i]),
				currentIter, m_imagePaths.size());
		}
	}
	
	for(unsigned int i = 0; i < histograms.size(); i++)
		delete features[i];
			
	return histograms;
}

double ClassificationFramework::trainClassifier() {
	vector<Histogram*> histograms = generateHistograms();

	vector<unsigned int> imageClasses = m_datasetManager->getImageClasses();
	vector<string> classNames = m_datasetManager->listClasses();
	
	Classifier classifier(histograms, imageClasses, classNames,
		m_settings.trainImagesPerClass);
	classifier.classify(m_settings.C);
	double result = classifier.test();
	
	for(unsigned int i = 0; i < histograms.size(); i++)
		delete histograms[i];

	return result;
}

double ClassificationFramework::run() {	
	return trainClassifier();
}
