#include "Settings.h"
using namespace std;
using boost::property_tree::ptree;

Settings::Settings() {
	colourspace = Image::GREYSCALE;
	featureType = FeatureExtractor::DSIFT;
	gridSpacing = 8;
	patchSize = 16;
	textonImages = 50;
	codewords = 200;
	pyramidLevels = 2;
	C = 10.0;
	trainImagesPerClass = 100;
}

Settings::Settings(std::string filename) {
	ptree tree;
	read_xml(filename, tree);
	
	datasetPath = tree.get<string>("settings.dataset");
	
	colourspace =
		(Image::Colourspace)tree.get("settings.features.colourspace", 0);
	featureType =
		(FeatureExtractor::Type)tree.get("settings.features.type", 0);
	
	gridSpacing = tree.get("settings.features.gridspacing", 8);
	patchSize = tree.get("settings.features.patchsize", 16);
	
	textonImages = tree.get("settings.codebook.textonimages", 50);
	codewords = tree.get("settings.codebook.codewords", 200);
	
	pyramidLevels = tree.get("settings.histograms.pyramidlevels", 2);
	
	C = tree.get("settings.classifier.c", 10);
	trainImagesPerClass = tree.get("settings.classifier.imagesperclass", 100);
}
