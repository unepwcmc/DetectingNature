#ifndef TRAINING_SETTINGS_H
#define TRAINING_SETTINGS_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "OutputHelper.h"
#include "DatasetManager.h"

class TrainingSettings {
public:
	TrainingSettings();
	
	void setDatasetSettings(std::string datasetName,
		unsigned int numTrainingImages);
	
	void setCodebookSettings(unsigned int codebookSize, 
		std::string featureDetectorName, std::string descriptorExtractorName);
	
	void setHistogramSettings(std::string descriptorMatcherName);
	
	unsigned int getCodebookSize() const;
	const OutputHelper* getOutputHelper() const;
	const DatasetManager* getDatasetManager() const;
	cv::Ptr<cv::FeatureDetector> getFeatureDetector() const;
	cv::Ptr<cv::DescriptorExtractor> getDescriptorExtractor() const;
	cv::Ptr<cv::DescriptorMatcher> getDescriptorMatcher() const;

private:
	// General helpers
	OutputHelper* m_outputHelper;
	DatasetManager* m_datasetManager;
	
	// Codebook generation settings
	unsigned int m_codebookSize;
	std::string m_featureDetectorName;
	cv::Ptr<cv::FeatureDetector> m_featureDetector;
	std::string m_descriptorExtractorName;
	cv::Ptr<cv::DescriptorExtractor> m_descriptorExtractor;

	// Histogram creator settings
	std::string m_descriptorMatcherName;
	cv::Ptr<cv::DescriptorMatcher> m_descriptorMatcher;
};

#endif
