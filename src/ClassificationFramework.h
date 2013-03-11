#ifndef CLASSIFICATION_FRAMEWORK_H
#define CLASSIFICATION_FRAMEWORK_H

#include <fstream>

#include "CacheHelper.h"
#include "DatasetManager.h"
#include "FeatureExtractor.h"
#include "CodebookGenerator.h"
#include "Classifier.h"
#include "Settings.h"

class ClassificationFramework {
public:
	ClassificationFramework(Settings &settings);
	~ClassificationFramework();
	
	double run();	

private:
	Settings m_settings;
	CacheHelper* m_cacheHelper;
	DatasetManager* m_datasetManager;
	std::vector<std::string> m_imagePaths;
	std::string m_cachePath;
	
	std::vector<ImageFeatures*> generateFeatures(
		std::vector<std::string> imagePaths);
	std::vector<Histogram*> generateHistograms(
		std::vector<std::string> imagePaths);
	double trainClassifier();
};

#endif
