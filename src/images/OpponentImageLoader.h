#ifndef OPPONENT_IMAGE_H
#define OPPONENT_IMAGE_H

#include "images/ImageLoader.h"

/**
 * @brief Loads the raw data of images in the Opponent colourspace.
 *
 * The class will also resize the image when it exceeds a predefined resolution
 * in order to reduce the computational costs of processing the image.
 */
class OpponentImageLoader : public ImageLoader {
public:
	/**
	 * @brief Initializes the image loader settings.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	OpponentImageLoader(const SettingsManager* settings);
	
private:
	ImageData* processImageData(cimg_library::CImg<float> image) const;
};

#endif
