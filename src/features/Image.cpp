#include "Image.h"
#include "iostream"
using namespace std;
using namespace cimg_library;

Image::Image(std::string filename, Colourspace colour) {
	CImg<float> image = CImg<float>(filename.c_str());
	if(colour == GREYSCALE && image.spectrum() > 1) {
		image = image.RGBtoHSI().get_channel(2);
	}
	
	int maxSize = max(image.height(), image.width());
	if(maxSize > 1000) {
		int resizeFactor = -100 * (1000.0 / maxSize);
		image = image.resize(resizeFactor, resizeFactor, -100, -100, 5);
	}

	m_width = image.width();
	m_height = image.height();

	for(int i = 0; i < image.spectrum(); i++) {
		m_data.push_back(new float[m_width * m_height]);
	}
	
	for(unsigned int y = 0; y < m_height; y++) {
		for(unsigned int x = 0; x < m_width; x++) {
			if(colour == OPPONENT) {
				float r, g, b;
				r = image(x, y, 0, 0);
				g = image(x, y, 0, 1);
				b = image(x, y, 0, 2);
				
				m_data[0][y * m_width + x] = 0.5 * (255.0 + g - r);
				m_data[1][y * m_width + x] =
					0.25 * (510.0 + r + g - (2 * b));
				m_data[2][y * m_width + x] =
					1.0 / 3.0 * (r + g + b);
			} else if(colour == HSV) {
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
				m_data[0][y * m_width + x] = 255.0 * v;
				m_data[1][y * m_width + x] = 255.0 * s;
				m_data[2][y * m_width + x] = h / 2.0;
			} else {
				m_data[0][y * m_width + x] = image(x, y);
			}
		}
	}
}

Image::~Image() {
	for(unsigned int i = 0; i < m_data.size(); i++) {
		delete[] m_data[i];
	}
}
