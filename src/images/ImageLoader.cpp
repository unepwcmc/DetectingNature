#include "ImageLoader.h"
#include "iostream"
using namespace std;
using namespace cimg_library;

ImageLoader::ImageLoader(const SettingsManager* settings) {
	cimg::imagemagick_path("/usr/bin/convert");
	m_maxRes = settings->get<double>("image.maxResolution");
}

ImageLoader::~ImageLoader() {
}

ImageData* ImageLoader::loadImage(std::string filename) const {
	CImg<float> image = CImg<float>(filename.c_str());
	
	int maxSize = max(image.height(), image.width());
	if(maxSize > m_maxRes) {
		int resizeFactor = -100 * (m_maxRes / maxSize);
		image = image.resize(resizeFactor, resizeFactor, -100, -100, 5);
	}
	
	return processImageData(image);
}
