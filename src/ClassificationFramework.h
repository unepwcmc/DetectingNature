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
	ClassificationFramework(std::string datasetPath,
		Settings &settings, bool skipCache);
	~ClassificationFramework();
	
	double run();	

private:
	bool m_skipCache;
	Settings m_settings;
	CacheHelper* m_cacheHelper;
	DatasetManager* m_datasetManager;
	std::vector<std::string> m_imagePaths;
	std::string m_cachePath;
	
	std::vector<ImageFeatures*> generateFeatures(
		std::vector<std::string> imagePaths);
	std::vector<Histogram*> generateHistograms(
		std::vector<std::string> imagePaths, bool skipCodebook);
	double trainClassifier();
};

#endif
