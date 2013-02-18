#include "CodebookManager.h"
using namespace std;
using namespace boost;

CodebookManager::CodebookManager(const TrainingSettings* settings) {
	
	m_settings = settings;
	m_outputHelper = settings->getOutputHelper();
	m_codebookTrainer = nullptr;
}

string CodebookManager::getCacheFilename() {
	return "vocab.xml";
}

void CodebookManager::generateMissingVocabulary() {
	m_outputHelper->printMessage("Extracting Features:");
	
	string cacheFilename = getCacheFilename();
	if(filesystem::exists(cacheFilename)) {
		cv::FileStorage fs(cacheFilename, cv::FileStorage::READ);
		fs["vocabulary"] >> m_vocabulary;
		
		m_outputHelper->printMessage("Loaded from " + cacheFilename, 1);
		return;
	}
	
	// Initialize Cluster Trainer
	m_codebookTrainer = new cv::BOWKMeansTrainer(m_settings->getCodebookSize());

	// Go trough all classes in the given dataset
	const vector<string> filenames =
		m_settings->getDatasetManager()->listFiles(DatasetManager::TRAIN);
	
	unsigned int totalProcessedImages = 0;
	
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < filenames.size(); i++) {
		string img = filenames[i];
		
		// Load the image
		cv::Mat input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);
				
		// Extract features
		std::vector<cv::KeyPoint> keypoints;
		m_settings->getFeatureDetector()->detect(input, keypoints);
		
		cv::Mat features;
		m_settings->getDescriptorExtractor()->compute(input,
			keypoints, features);
		
		// Train bag-of-words
		#pragma omp critical
		{
			totalProcessedImages++;
		
	    	m_codebookTrainer->add(features);
	    
		    // Show progress
		    m_outputHelper->printProgress("Processing file " +
		    	DatasetManager::getFilename(img),
		    	totalProcessedImages, filenames.size());
		}
	}
	
	computeCodebook();
}

void CodebookManager::computeCodebook() {
	m_outputHelper->printMessage("Computing Codebook:");
	m_outputHelper->printInlineMessage(str(format("Clustering %1% descriptors") %
		m_codebookTrainer->descripotorsCount()), 1);

	m_vocabulary = m_codebookTrainer->cluster();
	
	cv::FileStorage fs(getCacheFilename(), cv::FileStorage::WRITE);
	fs << "vocabulary" << m_vocabulary;
	
	m_outputHelper->printMessage("Done", 1);
}

cv::Mat CodebookManager::getVocabulary() {
	return m_vocabulary;
}

