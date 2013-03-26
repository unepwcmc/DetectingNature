#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>

#define cimg_display 0
#include <CImg.h>

/**
 * @brief Loads and stores the raw data of one image.
 *
 * The class will also transform the image into the desired colourspace for
 * feature extraction.
 */
class Image {
public:
	/**
	 * @brief The colourspace transformation to apply to the image.
	 */
	enum Colourspace {
		GREYSCALE, /**< Uses a single, grey channel */
		OPPONENT, /**< Converts the image into 3 opponent colouspace channels */
		HSV /**< Converts the image into 3 channels representing HSV */
	};

	/**
	 * @brief Initializes and converts the image data.
	 *
	 * @warning Processing the image in colour triples the required memory
	 * and hard drive storage.
	 *
	 * @param filename The location of the file to be loaded.
	 * @param colour The colourspace transformation to apply to the data.
	 */
	Image(std::string filename, Colourspace colour);
	~Image();
	
	/**
	 * @brief Returns the number of channels in the loaded image
	 * 
	 * This is usually one for greyscale images and three for color images.
	 *
	 * @return The number of data channels in the image.
	 */
	unsigned int getNumChannels() const {
		return m_data.size();
	}
	
	/**
	 * @brief Returns the width of each channel of the loaded image.
	 *
	 * @return The image width.
	 */
	unsigned int getWidth() const {
		return m_width;
	}
	
	/**
	 * @brief Returns the height of each channel of the loaded image.
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
