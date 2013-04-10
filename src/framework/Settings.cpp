#include "Settings.h"
using namespace std;
using boost::property_tree::ptree;

Settings::Settings() {
	colourspace = Image::GREYSCALE;
	featureType = FeatureExtractor::DSIFT;
	smoothingSigma = 0;
	gridSpacing = 8;
	patchSize = 16;
	textonImages = 50;
	codewords = 200;
	histogramType = Codebook::SQUARES;
	pyramidLevels = 2;
	C = 10.0;
	trainImagesPerClass = 100;
}

Settings::Settings(std::string filename) {
	ptree tree;
	read_json(filename, tree);
	
	colourspace =
		(Image::Colourspace)tree.get("features.colourspace", 0);
	featureType =
		(FeatureExtractor::Type)tree.get("features.type", 0);
	
	smoothingSigma = tree.get("features.smoothingsigma", 0);
	gridSpacing = tree.get("features.gridspacing", 8);
	patchSize = tree.get("features.patchsize", 16);
	
	textonImages = tree.get("codebook.textonimages", 50);
	codewords = tree.get("codebook.codewords", 200);
	
	histogramType = (Codebook::Type)tree.get("histograms.type", 0);
	pyramidLevels = tree.get("histograms.pyramidlevels", 2);
	
	C = tree.get("classifier.c", 10.0);
	trainImagesPerClass = tree.get("classifier.imagesperclass", 100);
}
