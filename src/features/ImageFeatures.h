#ifndef IMAGE_FEATURES_H
#define IMAGE_FEATURES_H

#include <cstring>

#include <boost/serialization/vector.hpp>

/**
 * @brief Stores all the features of one image.
 *
 * When an image contains multiple channels, the descriptors for each channel
 * are merged, creating a longer descriptor for each point.
 */
class ImageFeatures {
public:
	/**
	 * @brief Defines some information reguarding the original image.
	 *
	 * This information is required when computing the spatial histograms, to
	 * divide the image into equal partitions.
	 *
	 * @param width The width of the original image.
	 * @param height The height of the original image.
	 * @param numChannels The number of channels of the original image.
	 */
	ImageFeatures(unsigned int width, unsigned int height,
		unsigned int numChannels);
	
	/**
	 * @brief Stores the extracted features for an image.
	 *
	 * If the image contains more than one channel, the features will be
	 * appended, creating a larger descriptor.
	 *
	 * @warning Until this function has been called for all channels, the
	 * arrays returned by getFeature() and getFeatures() may contain
	 * invalid data.
	 * 
	 * @param channel The image channel where these features were
	 * extracted from.
	 * @param features An array with all the extracted features for this
	 * channel.
	 * @param descriptorSize The length of each descriptor.
	 * @param numFeatures The number of features in the @a features array.
	 * @param coordinates The keypoint where the features were extracted.
	 * These must be presented in the same order as the @a features array.
	 */

		
	void extendFeatures(unsigned int channel, float const* features,
		unsigned int numFeatures);
	void newFeatures(float const* features,
		unsigned int descriptorSize, unsigned int numFeatures,
		std::vector<std::pair<int, int> > coordinates);
	
	/**
	 * @brief Returns the total number of stored features.
	 *
	 * This value does not change based on the number of channels, since more
	 * channels mean longer descriptors and not more descriptors.
	 *
	 * @return The number of stored features.
	 */
	unsigned int getNumFeatures() const {
		return m_numFeatures;
	}
	
	/**
	 * @brief Returns the size of each descriptor
	 *
	 * This size already takes into account the number of channels of the image.
	 *
	 * @return The size of each descriptor.
	 */
	unsigned int getDescriptorSize() const {
		return m_descriptorSize;
	}
	
	/**
	 * @brief Fetches a single feature
	 *
	 * The returned array is getDescriptorSize() long.
	 *
	 * @param index A zero based index for the requested feature.
	 * @return A pointer to the requested feature.
	 */
	const float* getFeature(unsigned int index) const {
		return &m_features[index * m_descriptorSize];
	}
	
	/**
	 * @brief Fetches all the features of the image.
	 *
	 * The returned array is getNumFeatures() times getDescriptorSize() long.
	 * 
	 * @return An array containing the entire feature set.
	 */
	const float* getFeatures() const {
		return &m_features[0];
	}
	
	/**
	 * @brief Returns the coordinates of one feature.
	 * 
	 * @param index A zero based index of the feature to get the
	 * coordinates from.
	 * @return A pair containing the @a X coordinate in the first element and
	 * the @a Y coordinate in the second element.
	 */
	std::pair<int, int> getCoordinates(unsigned int index) const {
		return m_coordinates[index];
	}
	
	/**
	 * @brief Returns the width of the original image.
	 *
	 * This is required to evenly distribute the values returned
	 * by getCoordinates().
	 *
	 * @return The width of the image.
	 */
	unsigned int getWidth() const {
		return m_width;
	}
	
	/**
	 * @brief Returns the height of the original image.
	 *
	 * This is required to evenly distribute the values returned
	 * by getCoordinates().
	 *
	 * @return The height of the image.
	 */
	unsigned int getHeight() const {
		return m_height;
	}
	
private:
	unsigned int m_numChannels;
	unsigned int m_descriptorSize;
	unsigned int m_numFeatures;
	std::vector<float> m_features;
	unsigned int m_width;
	unsigned int m_height;
	std::vector<std::pair<int, int> > m_coordinates;
	
	// Boost serialization
	friend class boost::serialization::access;
	ImageFeatures();
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_numChannels;
		ar & m_descriptorSize;
		ar & m_numFeatures;
		ar & m_features;
		ar & m_width;
		ar & m_height;
		ar & m_coordinates;
	}
};

#endif
