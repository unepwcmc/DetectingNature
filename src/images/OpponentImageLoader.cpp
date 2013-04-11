#include "OpponentImageLoader.h"
#include "iostream"
using namespace std;
using namespace cimg_library;

OpponentImageLoader::OpponentImageLoader(const SettingsManager* settings) :
	ImageLoader(settings) {
}

ImageData* OpponentImageLoader::processImageData(CImg<float> image) const {
	std::vector<float*> data;
	unsigned int width = image.width();
	unsigned int height = image.height();
	
	for(int i = 0; i < 3; i++) {
		data.push_back(new float[width * height]);
	}
	
	for(unsigned int y = 0; y < height; y++) {
		for(unsigned int x = 0; x < width; x++) {
			float r, g, b;
			r = image(x, y, 0, 0);
			g = image(x, y, 0, 1);
			b = image(x, y, 0, 2);
			
			data[0][y * width + x] = 0.5 * (255.0 + g - r);
			data[1][y * width + x] =
				0.25 * (510.0 + r + g - (2 * b));
			data[2][y * width + x] =
				1.0 / 3.0 * (r + g + b);
		}
	}
	return new ImageData(data, width, height);
}
