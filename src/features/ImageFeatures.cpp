#include "ImageFeatures.h"
using namespace std;

ImageFeatures::ImageFeatures() {
	m_numChannels = 0;
	m_descriptorSize = 0;
	m_numFeatures = 0;
	m_width = 0;
	m_height = 0;
};

ImageFeatures::ImageFeatures(unsigned int width, unsigned int height,
		unsigned int numChannels) {
		
	m_numChannels = numChannels;
	m_descriptorSize = 0;
	m_numFeatures = 0;
	m_width = width;
	m_height = height;
}

void ImageFeatures::extendFeatures(unsigned int channel, float const* features,
		unsigned int numFeatures) {
	
	unsigned int baseIndex = m_features.size();
	for(unsigned int i = 0; i < m_descriptorSize * numFeatures; i++) {
		m_features[((baseIndex + i) * m_numChannels) + channel] = features[i];
	}
}

void ImageFeatures::newFeatures(float const* features,
		unsigned int descriptorSize, unsigned int numFeatures,
		vector<pair<int, int> > coordinates) {
	
	m_descriptorSize = m_numChannels * descriptorSize;
	m_numFeatures += numFeatures;
	
	m_coordinates.insert(m_coordinates.end(),
		coordinates.begin(), coordinates.end());
	
	unsigned int baseIndex = m_features.size();
	m_features.resize(m_descriptorSize * m_numFeatures);
	for(unsigned int i = 0; i < descriptorSize * numFeatures; i++) {
		m_features[((baseIndex + i) * m_numChannels)] = features[i];
	}
}
