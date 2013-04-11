#include "ImageData.h"
using namespace std;

ImageData::ImageData(vector<float*> data,
		unsigned int width, unsigned int height) {
		
	m_data = data;
	m_width = width;
	m_height = height;
}

ImageData::~ImageData() {
	for(unsigned int i = 0; i < m_data.size(); i++) {
		delete[] m_data[i];
	}
}
