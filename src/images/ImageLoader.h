#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <string>

#define cimg_display 0
#include <CImg.h>

#include "images/ImageData.h"
#include "framework/SettingsManager.h"

/**
 * @brief Loads the raw data of images.
 *
 * The class will also resize the image when it exceeds a predefined resolution
 * in order to reduce the computational costs of processing the image.
 */
class ImageLoader {
public:
	/**
	 * @brief Initializes the image loader settings.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	ImageLoader(const SettingsManager* settings);
	virtual ~ImageLoader();
	
		/**
	 * @brief Loads the image data.
	 *
	 * @warning Processing an image in colour triples the required memory
	 * and hard drive storage.
	 *
	 * @param filename The location of the file to be loaded.
	 * @return The data of the resized and processed image.
	 */
	ImageData* loadImage(std::string filename) const;

private:
	double m_maxRes;
	bool m_forceSize;
	
	virtual ImageData* processImageData(
		cimg_library::CImg<float> image) const = 0;
};

#endif
