#include "ImageFeatures.h"

ImageFeatures::ImageFeatures() {
};

ImageFeatures::ImageFeatures(float const* features,
		unsigned int descriptorSize, unsigned int numFeatures) {
	
	m_features.reserve(descriptorSize * numFeatures);
	
	copy(features, features + (descriptorSize * numFeatures),
		back_inserter(m_features));
	
	m_descriptorSize = descriptorSize;
	m_numFeatures = numFeatures;
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
