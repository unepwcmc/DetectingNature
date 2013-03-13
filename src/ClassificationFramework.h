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
	
	double testRun();
	std::map<std::string, std::string> classify(std::string imagesFolder);

private:
	bool m_skipCache;
	Settings m_settings;
	CacheHelper* m_cacheHelper;
	DatasetManager* m_datasetManager;
	FeatureExtractor* m_featureExtractor;
	std::vector<std::string> m_imagePaths;
	std::string m_cachePath;
	
	Codebook* prepareCodebook(
		std::vector<std::string> imagePaths, bool skipCache);
	ImageFeatures* extractFeature(std::string imagePath);
	Histogram* generateHistogram(Codebook* codebook, std::string filePath);
	Classifier* trainClassifier(std::vector<Histogram*> trainHistograms); 
	
	std::vector<ImageFeatures*> extractFeatures(
		std::vector<std::string> imagePaths);
	std::vector<Histogram*> generateHistograms(
		std::vector<std::string> imagePaths, bool skipCodebook);
	double trainClassifier();
};

#endif
