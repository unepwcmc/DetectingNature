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
	struct Settings {
		std::string datasetPath;
		
		// Feature settings
		FeatureExtractor::Type featureType = FeatureExtractor::DSIFT;
		unsigned int gridSpacing = 8;
		unsigned int patchSize = 16;
		
		// Codebook settings
		unsigned int textonImages = 50;
		unsigned int codewords = 200;
		
		// Histogram settings
		unsigned int pyramidLevels = 2;
		
		// Classifier settings
		double C = 10.0;
		unsigned int trainImagesPerClass = 100;
	};

	ClassificationFramework(Settings &settings);
	~ClassificationFramework();
	
	double run();	

private:
	Settings m_settings;
	DatasetManager* m_datasetManager;
	std::vector<std::string> m_imagePaths;
	std::string m_cachePath;
	
	std::vector<ImageFeatures*> generateFeatures();
	std::vector<Histogram*> generateHistograms();
	double trainClassifier();
};

#endif
