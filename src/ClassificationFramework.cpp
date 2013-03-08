#include "ClassificationFramework.h"
using namespace std;

ClassificationFramework::ClassificationFramework(Settings &settings) {
	m_settings = settings;

	m_cachePath = "cache/" + settings.datasetPath;
	if(!boost::filesystem::exists(m_cachePath)) {
		boost::filesystem::create_directories(m_cachePath);
	}
	
	string datasetCache = m_cachePath + "/dataset";
	if(!boost::filesystem::exists(datasetCache)) {
		m_datasetManager = new DatasetManager(settings.datasetPath);
		
		ofstream ofs(datasetCache);
		boost::archive::binary_oarchive oa(ofs);
		oa << m_datasetManager;
	} else {
		ifstream ifs(datasetCache);
		boost::archive::binary_iarchive ia(ifs);
		ia >> m_datasetManager;
	}
	
	m_imagePaths = m_datasetManager->listFiles();
}

ClassificationFramework::~ClassificationFramework() {
	delete m_datasetManager;
}

vector<ImageFeatures*> ClassificationFramework::generateFeatures() {
	OutputHelper::printMessage("Extracting features:");
	
	stringstream cacheNameStream;
	cacheNameStream	<< 
		"_" << m_settings.colourspace <<
		"_" << m_settings.featureType <<
		"_" << m_settings.gridSpacing << "_" << m_settings.patchSize;
	string cacheFolder = m_cachePath + "/descriptors" + cacheNameStream.str();
	if(!boost::filesystem::exists(cacheFolder)) {
		boost::filesystem::create_directories(cacheFolder);
	}
	
	FeatureExtractor featureExtractor(m_settings.featureType,
		m_settings.gridSpacing, m_settings.patchSize);
	vector<ImageFeatures*> features(m_imagePaths.size(), nullptr);	

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_imagePaths.size(); i++) {
		string cacheFilename = cacheFolder + "/" +
			boost::replace_all_copy(m_imagePaths[i], "/", "_");
		if(!boost::filesystem::exists(cacheFilename)) {
			Image img(m_imagePaths[i], m_settings.colourspace);
			features[i] = featureExtractor.extract(img);
			
			ofstream ofs(cacheFilename);
			boost::archive::binary_oarchive oa(ofs);
			oa << features[i];
		} else {
			ifstream ifs(cacheFilename);
			boost::archive::binary_iarchive ia(ifs);
			ia >> features[i];
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
	stringstream cacheNameStream;
	cacheNameStream	<< "_" << m_settings.textonImages <<
		"_" << m_settings.codewords << "_" << m_settings.pyramidLevels;
	string cacheFolder = m_cachePath + "/histograms" + cacheNameStream.str();
	if(!boost::filesystem::exists(cacheFolder)) {
		boost::filesystem::create_directories(cacheFolder);
	}
	
	vector<Histogram*> histograms(m_imagePaths.size(), nullptr);
	
	vector<ImageFeatures*> features = generateFeatures(); 
	
	OutputHelper::printMessage("Generating histograms:");
			
	CodebookGenerator codebookGenerator(features);
	Codebook* codebook = codebookGenerator.generate(
		m_settings.textonImages, m_settings.codewords);

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_imagePaths.size(); i++) {
		string cacheFilename = cacheFolder + "/" +
			boost::replace_all_copy(m_imagePaths[i], "/", "_");
			
		if(!boost::filesystem::exists(cacheFilename)) {
			histograms[i] = codebook->computeHistogram(features[i],
				m_settings.pyramidLevels);
				
			ofstream ofs(cacheFilename);
			boost::archive::binary_oarchive oa(ofs);
			oa << histograms[i];
		} else {
			ifstream ifs(cacheFilename);
			boost::archive::binary_iarchive ia(ifs);
			ia >> histograms[i];
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
