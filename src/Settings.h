#ifndef SETTINGS_H
#define SETTINGS_H

#include "Image.h"
#include "FeatureExtractor.h"

struct Settings {
	std::string datasetPath;
	
	// Image settings
	Image::Colourspace colourspace = Image::GREYSCALE;
	
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

#endif
