#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

#include <vector>

/**
 * @brief Stores the raw data of one image.
 *
 */
class ImageData {
public:
	/**
	 * @brief Initializes the image data.
	 *
	 * @warning The data is not copied. Do not delete it after calling this
	 * constructor.
	 *
	 * @param data A vector containing an array of pixel data for each
	 * dimension.
	 * @param width Width of the image
	 * @param height Height of the image
	 */
	ImageData(std::vector<float*> data,
		unsigned int width, unsigned int height);
	~ImageData();
	
	/**
	 * @brief Returns the number of channels in the image
	 * 
	 * This is usually one for greyscale images and three for color images.
	 *
	 * @return The number of data channels in the image.
	 */
	unsigned int getNumChannels() const {
		return m_data.size();
	}
	
	/**
	 * @brief Returns the width of each channel of the image.
	 *
	 * @return The image width.
	 */
	unsigned int getWidth() const {
		return m_width;
	}
	
	/**
	 * @brief Returns the height of each channel of the image.
	 *
	 * @return The image height.
	 */
	unsigned int getHeight() const {
		return m_height;
	}
	
	/**
	 * @brief Fetches the raw image data of one single channel.
	 *
	 * @param channel The channel for which to retrieve the data. Must be
	 * lower than the value returned by getNumChannels()
	 * @return The image data.
	 */
	float const* getData(unsigned int channel) const {
		return m_data[channel];
	}

private:
	unsigned int m_width;
	unsigned int m_height;
	std::vector<float*> m_data;
};

#endif
