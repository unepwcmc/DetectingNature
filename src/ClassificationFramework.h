#ifndef CLASSIFICATION_FRAMEWORK_H
#define CLASSIFICATION_FRAMEWORK_H

#include <fstream>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>

#include "DatasetManager.h"
#include "FeatureExtractor.h"
#include "CodebookGenerator.h"
#include "Classifier.h"

class ClassificationFramework {
public:
	ClassificationFramework(std::string datasetPath);
	~ClassificationFramework();
	
	void run();	

private:
	DatasetManager* m_datasetManager;
	std::vector<std::string> m_imagePaths;
	std::string m_cachePath;
	
	std::vector<ImageFeatures*> generateFeatures();
	std::vector<Histogram*> generateHistograms();
	void trainClassifier();
};

#endif
