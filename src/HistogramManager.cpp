#include "HistogramManager.h"
using namespace std;
using namespace boost;

HistogramManager::HistogramManager(const TrainingSettings* settings,
		const cv::Mat vocabulary) {
		
	m_settings = settings;
	m_outputHelper = settings->getOutputHelper();
	
	m_bowExtractor = new cv::BOWImgDescriptorExtractor(
		m_settings->getDescriptorExtractor(),
		m_settings->getDescriptorMatcher());
	m_bowExtractor->setVocabulary(vocabulary);
}

string HistogramManager::getCacheFilename() {
	return "histograms.xml";
}

string HistogramManager::getSimplifiedFilename(string filename) {
	replace(filename.begin(), filename.end(), '/', '-');
	replace(filename.begin(), filename.end(), '.', '-');
	
	return filename;
}

void HistogramManager::generateMissingHistograms() {
		
	m_outputHelper->printMessage("Generating histograms:");
		
	string cacheFilename = getCacheFilename();
	if(filesystem::exists(cacheFilename)) {
		m_outputHelper->printMessage("Loaded from " + cacheFilename, 1);
		return;
	}
		
	// Go trough all classes in the given dataset
	vector<string> filenames =
		m_settings->getDatasetManager()->listFiles(DatasetManager::ALL);
	
	cv::FileStorage fs(cacheFilename, cv::FileStorage::WRITE);

	unsigned int totalProcessedImages = 0;
	#pragma omp parallel for
	for(unsigned int i = 0;	i < filenames.size(); i++) {
		string img = filenames[i];
	
		// Extract features
		cv::Mat input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<cv::KeyPoint> keypoints;
		m_settings->getFeatureDetector()->detect(input, keypoints);
		
		// Compute image final descriptor
		cv::Mat img_descriptor;
	    m_bowExtractor->compute(input, keypoints, img_descriptor);
	    
	    #pragma omp critical
	    {	    
		    totalProcessedImages++;
		    
		    string imgName = getSimplifiedFilename(img);
			fs << imgName << img_descriptor;

			// Show progress
			m_outputHelper->printProgress("Processing file " +
				DatasetManager::getFilename(img),
				totalProcessedImages, filenames.size(), 2);
		}
	}
}

