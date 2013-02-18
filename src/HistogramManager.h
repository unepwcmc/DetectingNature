#ifndef HISTOGRAM_MANAGER_H
#define HISTOGRAM_MANAGER_H

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>

#include "TrainingSettings.h"

class HistogramManager {
public:
	HistogramManager(const TrainingSettings* settings,
		const cv::Mat vocabulary);
	void generateMissingHistograms();
	
private:
	const TrainingSettings* m_settings;
	const OutputHelper* m_outputHelper;
	cv::BOWImgDescriptorExtractor* m_bowExtractor;
	
	std::string getCacheFilename();
	std::string getSimplifiedFilename(std::string filename);
};

#endif
