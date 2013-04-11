#include "GreyscaleImageLoader.h"
using namespace std;
using namespace cimg_library;

GreyscaleImageLoader::GreyscaleImageLoader(const SettingsManager* settings) :
	ImageLoader(settings) {
}

ImageData* GreyscaleImageLoader::processImageData(CImg<float> image) const {
	std::vector<float*> data;
	unsigned int width = image.width();
	unsigned int height = image.height();
	
	data.push_back(new float[width * height]);
	
	for(unsigned int y = 0; y < height; y++) {
		for(unsigned int x = 0; x < width; x++) {
			data[0][y * width + x] = image(x, y);
		}
	}
	return new ImageData(data, width, height);
}
