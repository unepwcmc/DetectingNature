#include "Image.h"
#include "iostream"
using namespace std;

Image::Image(std::string filename, Colourspace colour) {
	cv::Mat cvImg = cv::imread(filename, colour != GREYSCALE);

	m_width = cvImg.rows;
	m_height = cvImg.cols;

	unsigned int numChannels = colour == GREYSCALE ? 1 : 3;
	for(unsigned int i = 0; i < numChannels; i++) {
		m_data.push_back(new float[m_width * m_height]);
	}
	
	cv::Point3_<unsigned char> point;
	for(unsigned int y = 0; y < m_height; y++) {
		for(unsigned int x = 0; x < m_width; x++) {
			switch(colour) {
			case OPPONENT:
				point =	cvImg.at<cv::Point3_<unsigned char> >(x, y);
				m_data[0][y * m_width + x] = (point.z - point.y) / sqrt(2);
				m_data[1][y * m_width + x] =
					(point.z + point.y - (2 * point.x)) / sqrt(6);
				m_data[2][y * m_width + x] =
					(point.z + point.y + point.x) / sqrt(3);
				break;
			case GREYSCALE:
			default:
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

unsigned int Image::getNumChannels() const {
	return m_data.size();
}

unsigned int Image::getWidth() const {
	return m_width;
}

unsigned int Image::getHeight() const {
	return m_height;
}

float const* Image::getData(unsigned int channel) const {
	return m_data[channel];
}
