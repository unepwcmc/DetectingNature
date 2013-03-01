#include "ClassificationFramework.h"
using namespace std;

ClassificationFramework::ClassificationFramework(string datasetPath) {
	m_cachePath = "cache/" + datasetPath;
	if(!boost::filesystem::exists(m_cachePath)) {
		boost::filesystem::create_directories(m_cachePath);
	}
	
	string datasetCache = m_cachePath + "/dataset";
	if(!boost::filesystem::exists(datasetCache)) {
		m_datasetManager = new DatasetManager(datasetPath);
		
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
	
	string cacheFilename = m_cachePath + "/descriptors";

	FeatureExtractor featureExtractor;
	vector<ImageFeatures*> features(m_imagePaths.size(), nullptr);

	if(!boost::filesystem::exists(cacheFilename)) {
		unsigned int currentIter = 0;
		#pragma omp parallel for
		for(unsigned int i = 0; i < m_imagePaths.size(); i++) {		
			Image img(m_imagePaths[i]);
			ImageFeatures* imageFeatures = featureExtractor.extractDsift(img);
			features[i] = imageFeatures;
					
			#pragma omp critical
			{
				currentIter++;
				OutputHelper::printProgress("Processing image "
					+ DatasetManager::getFilename(m_imagePaths[i]),
					currentIter, m_imagePaths.size());
			}
		}

		ofstream ofs(cacheFilename);
		boost::archive::binary_oarchive oa(ofs);
		oa << features;
	} else {
		OutputHelper::printInlineMessage("Loading from " + cacheFilename, 1);
		ifstream ifs(cacheFilename);
		boost::archive::binary_iarchive ia(ifs);
		ia >> features;
		OutputHelper::printMessage("Loaded from " + cacheFilename, 1);
	}
	
	return features;
}

vector<Histogram*> ClassificationFramework::generateHistograms() {
	string cacheFilename = m_cachePath + "/histograms";
	
	vector<Histogram*> histograms(m_imagePaths.size(), nullptr);
	
	if(!boost::filesystem::exists(cacheFilename)) {
		vector<ImageFeatures*> features = generateFeatures();
		
		OutputHelper::printMessage("Generating histograms:");
				
		CodebookGenerator codebookGenerator(features);
		Codebook* codebook = codebookGenerator.generate(50, 200);
	
		unsigned int currentIter = 0;
		#pragma omp parallel for
		for(unsigned int i = 0; i < m_imagePaths.size(); i++) {
			Histogram* hist = codebook->computeHistogram(features[i]);
			histograms[i] = hist;
			
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
		
		ofstream ofs(cacheFilename);
		boost::archive::binary_oarchive oa(ofs);
		oa << histograms;
	} else {
		OutputHelper::printMessage("Generating histograms:");
		OutputHelper::printInlineMessage("Loading from " + cacheFilename, 1);
		ifstream ifs(cacheFilename);
		boost::archive::binary_iarchive ia(ifs);
		ia >> histograms;
		OutputHelper::printMessage("Loaded from " + cacheFilename, 1);
	}
	
	return histograms;
}

void ClassificationFramework::trainClassifier() {
	vector<Histogram*> histograms = generateHistograms();

	vector<unsigned int> imageClasses = m_datasetManager->getImageClasses();
	vector<string> classNames = m_datasetManager->listClasses();
	
	Classifier classifier(histograms, imageClasses, classNames, 100);
	classifier.classify();
	classifier.test();
	
	for(unsigned int i = 0; i < histograms.size(); i++)
		delete histograms[i];
}

void ClassificationFramework::run() {	
	trainClassifier();
}
