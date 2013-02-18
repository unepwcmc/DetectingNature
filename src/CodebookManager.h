#ifndef CODEBOOK_MANAGER_H
#define CODEBOOK_MANAGER_H

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "OutputHelper.h"
#include "TrainingSettings.h"

class CodebookManager {
public:
	CodebookManager(const TrainingSettings* settings);
	
	void generateMissingVocabulary();
	cv::Mat getVocabulary();
		
private:
	const TrainingSettings* m_settings;
	const OutputHelper* m_outputHelper;
	cv::Mat m_vocabulary;
	cv::BOWKMeansTrainer* m_codebookTrainer;
	
	void computeCodebook();
	std::string getCacheFilename();
};

#endif
