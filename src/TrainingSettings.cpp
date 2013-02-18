#include "TrainingSettings.h"
using namespace std;

TrainingSettings::TrainingSettings() {
	m_outputHelper = new OutputHelper();
}


void TrainingSettings::setDatasetSettings(string datasetName,
		unsigned int numTrainingImages) {

	m_datasetManager = new DatasetManager(datasetName, numTrainingImages);
}

void TrainingSettings::setCodebookSettings(unsigned int codebookSize, 
		string featureDetectorName, string descriptorExtractorName) {

	m_codebookSize = codebookSize;
	
	m_featureDetectorName = featureDetectorName;
	m_descriptorExtractorName = descriptorExtractorName;
	
	m_featureDetector = cv::FeatureDetector::create(featureDetectorName);
	m_descriptorExtractor =
		cv::DescriptorExtractor::create(descriptorExtractorName);
}

unsigned int TrainingSettings::getCodebookSize() const {
	return m_codebookSize;
}

cv::Ptr<cv::FeatureDetector> TrainingSettings::getFeatureDetector() const {
	return m_featureDetector;
}

cv::Ptr<cv::DescriptorExtractor> TrainingSettings::getDescriptorExtractor()
		const {

	return m_descriptorExtractor;
}

const OutputHelper* TrainingSettings::getOutputHelper() const {
	return m_outputHelper;
}

const DatasetManager* TrainingSettings::getDatasetManager() const {
	return m_datasetManager;
}
