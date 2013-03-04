#ifndef IMAGE_FEATURES_H
#define IMAGE_FEATURES_H

#include <cstring>

#include <boost/serialization/vector.hpp>

class ImageFeatures {
public:
	ImageFeatures();
	ImageFeatures(float const* features,
		unsigned int descriptorSize, unsigned int numFeatures,
		unsigned int width, unsigned int height,
		std::vector<std::pair<int, int> > coordinates);
	
	unsigned int getNumFeatures() const;
	unsigned int getDescriptorSize() const;
	const float* getFeature(unsigned int index) const;
	const float* getFeatures() const;
	std::pair<int, int> getCoordinates(unsigned int index) const;
	unsigned int getWidth() const;
	unsigned int getHeight() const;
	
private:
	unsigned int m_descriptorSize;
	unsigned int m_numFeatures;
	std::vector<float> m_features;
	unsigned int m_width;
	unsigned int m_height;
	std::vector<std::pair<int, int> > m_coordinates;
	
	// Boost serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_descriptorSize;
		ar & m_numFeatures;
		ar & m_features;
		ar & m_width;
		ar & m_height;
		ar & m_coordinates;
	}
};

#endif
