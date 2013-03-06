#include "ImageFeatures.h"
using namespace std;

ImageFeatures::ImageFeatures() {
	m_descriptorSize = 0;
	m_numFeatures = 0;
	m_width = 0;
	m_height = 0;
};

ImageFeatures::ImageFeatures(float const* features,
		unsigned int descriptorSize, unsigned int numFeatures,
		unsigned int width, unsigned int height,
		vector<pair<int, int> > coordinates) {
	
	m_features.reserve(descriptorSize * numFeatures);
	
	copy(features, features + (descriptorSize * numFeatures),
		back_inserter(m_features));
	
	m_descriptorSize = descriptorSize;
	m_numFeatures = numFeatures;
	m_width = width;
	m_height = height;
	m_coordinates = coordinates;
}

unsigned int ImageFeatures::getNumFeatures() const {
	return m_numFeatures;
}

unsigned int ImageFeatures::getDescriptorSize() const {
	return m_descriptorSize;
}

const float* ImageFeatures::getFeature(unsigned int index) const {
	return &m_features[index * m_descriptorSize];
}

const float* ImageFeatures::getFeatures() const {
	return &m_features[0];
}

pair<int, int> ImageFeatures::getCoordinates(unsigned int index) const {
	return m_coordinates[index];
}

unsigned int ImageFeatures::getWidth() const {
	return m_width;
}

unsigned int ImageFeatures::getHeight() const {
	return m_height;
}
