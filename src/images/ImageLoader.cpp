#include "ImageLoader.h"
#include "iostream"
using namespace std;
using namespace cimg_library;

ImageLoader::ImageLoader(const SettingsManager* settings) {
	cimg::imagemagick_path("/usr/bin/convert");
}

ImageLoader::~ImageLoader() {
}

ImageData* ImageLoader::loadImage(std::string filename) const {
	CImg<float> image = CImg<float>(filename.c_str());
	
	int maxSize = max(image.height(), image.width());
	if(maxSize > 1000) {
		int resizeFactor = -100 * (1000.0 / maxSize);
		image = image.resize(resizeFactor, resizeFactor, -100, -100, 5);
	}
	
	return processImageData(image);
}
