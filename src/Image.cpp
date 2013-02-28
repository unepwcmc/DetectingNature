#include "Image.h"
#include "iostream"
using namespace std;

Image::Image(std::string filename) {
	cv::Mat cvImg = cv::imread(filename, 0);

	m_width = cvImg.rows;
	m_height = cvImg.cols;

	m_data = new float[m_width * m_height];	
	for(unsigned int i = 0; i < m_height; i++) {
		for(unsigned int j = 0; j < m_width; j++) {
			m_data[i * m_width + j] = cvImg.at<unsigned char>(j, i);
		}
	}
}

Image::~Image() {
	delete m_data;
}

unsigned int Image::getWidth() const {
	return m_width;
}

unsigned int Image::getHeight() const {
	return m_height;
}

float const* Image::getData() const {
	return m_data;
}
