#include "HSVImageLoader.h"
using namespace std;
using namespace cimg_library;

HSVImageLoader::HSVImageLoader(const SettingsManager* settings) :
	ImageLoader(settings) {
}

ImageData* HSVImageLoader::processImageData(CImg<float> image) const {
	std::vector<float*> data;
	unsigned int width = image.width();
	unsigned int height = image.height();
	
	for(int i = 0; i < 3; i++) {
		data.push_back(new float[width * height]);
	}
	
	for(unsigned int y = 0; y < height; y++) {
		for(unsigned int x = 0; x < width; x++) {
			float r, g, b;
			r = image(x, y, 0, 0) / 255.0;
			g = image(x, y, 0, 1) / 255.0;
			b = image(x, y, 0, 2) / 255.0;
							
			float maxColour = max(r, max(g, b));
			float minColour = min(r, min(g, b));
			float chroma = maxColour - minColour;
			
			float v = maxColour;
			float s = (v == 0.0) ? 0.0 : chroma / v;
			float h = 0.0;
			
			// Ensure no divisions by zero occur
			if(chroma != 0) {
				if(v == r) {
					h = 60.0 * (g - b) / chroma;
				} else if(v == g) {
					h = 120.0 + 60.0 * (b - r) / chroma;
				} else {
					h = 240.0 + 60.0 * (r - g) / chroma;
				}
			}
			
			if(h < 0.0) {
				h += 360.0;
			}
			
			// Scale data into the range expected by VlFeat [0-255]
			data[0][y * width + x] = 255.0 * v;
			data[1][y * width + x] = 255.0 * s;
			data[2][y * width + x] = h / 2.0;
		}
	}
	return new ImageData(data, width, height);
}
