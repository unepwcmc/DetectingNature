#include "Image.h"
#include "iostream"
using namespace std;

Image::Image(std::string filename, Colourspace colour) {
	cv::Mat cvImg = cv::imread(filename, colour != GREYSCALE);

	m_width = cvImg.rows;
	m_height = cvImg.cols;

	unsigned int numChannels = (colour == GREYSCALE) ? 1 : 3;
	for(unsigned int i = 0; i < numChannels; i++) {
		m_data.push_back(new float[m_width * m_height]);
	}
	
	for(unsigned int y = 0; y < m_height; y++) {
		for(unsigned int x = 0; x < m_width; x++) {
			if(colour == OPPONENT) {
				cv::Point3_<unsigned char> point;
				point =	cvImg.at<cv::Point3_<unsigned char> >(x, y);

				m_data[0][y * m_width + x] = 0.5 * (255.0 + point.y - point.z);
				m_data[1][y * m_width + x] =
					0.25 * (510.0 + point.z + point.y - (2 * point.x));
				m_data[2][y * m_width + x] =
					1.0 / 3.0 * (point.z + point.y + point.x);
			} else if(colour == HSV) {
				cv::Point3_<unsigned char> point
					= cvImg.at<cv::Point3_<unsigned char> >(x, y);
				
				cv::Point3_<float> transformed;
				transformed.x = point.x / 255.0;
				transformed.y = point.y / 255.0;
				transformed.z = point.z / 255.0;
				
				float maxColour = max(transformed.x,
					max(transformed.y, transformed.z));
				float minColour = min(transformed.x,
					min(transformed.y, transformed.z));
				
				float chroma = maxColour - minColour;
				
				float v = maxColour;
				float s = (v == 0.0) ? 0.0 : chroma / v;
				float h = 0.0;
				
				// Ensure no divisions by zero occur
				if(chroma != 0) {
					if(v == transformed.z) {
						h = 60.0 * (transformed.y - transformed.x) / chroma;
					} else if(v == transformed.y) {
						h = 120.0 + 60.0 * (transformed.x - transformed.z)
							/ chroma;
					} else {
						h = 240.0 + 60.0 * (transformed.z - transformed.y)
							/ chroma;
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
				m_data[0][y * m_width + x] = cvImg.at<unsigned char>(x, y);
			}
		}
	}
}

Image::~Image() {
	for(unsigned int i = 0; i < m_data.size(); i++) {
		delete[] m_data[i];
	}
}