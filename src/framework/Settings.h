#ifndef SETTINGS_H
#define SETTINGS_H

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "features/Image.h"
#include "codebook/Codebook.h"
#include "features/FeatureExtractor.h"

/**
 * @brief Stores all the classification parameters.
 *
 * These are used to configure all the algorithms used in the image
 * classification process. They are also used to generate the cache folder
 * names.
 *
 * @see CacheHelper
 */
struct Settings {
	/**
	 * @brief Initializes a new settings structure.
	 *
	 * Uses the default values.
	 */
	Settings();
	
	/**
	 * @brief Loads a new settings structure.
	 * 
	 * Uses the values defined in an XML file.
	 */
	Settings(std::string filename);
	
	/**
	 * @brief The colourspace transformation to use when loading the image file.
	 * 
	 * @see Image::Colourspace
	 */
	Image::Colourspace colourspace;
	
	/**
	 * @brief The type of feature to use to describe an image.
	 * 
	 * @see FeatureExtractor::Type
	 */
	FeatureExtractor::Type featureType;
	
	/**
	 * @brief The sigma of the gaussian to be used on the smoothing/blurring
	 * aplied to the image.
	 *
	 * A value of zero disables smoothing completely.
	 */
	double smoothingSigma;
	
	/**
	 * @brief The spacing between each of the extracted descriptors.
	 * 
	 * The value is used for both the @a X and the @a Y axis.
	 */
	unsigned int gridSpacing;
	
	/**
	 * @brief The size of the descriptor patch.
	 *
	 * Usually a value higher than @a gridSpacing works better for
	 * classification.
	 *
	 * @warning Should be a multiple of four, otherwise it will be rounded
	 * to the closest multiple. 
	 */
	unsigned int patchSize;
	
	/**
	 * @brief Number of images to use to generate the Codebook.
	 *
	 * A random subset of @a textonImages images from the training dataset will be
	 * used to compute a Codebook using all of their features.
	 *
	 * @warning Being one of the slowest and most memory intensive steps of
	 * scene classification, a reasonably low number of images should be used.
	 * Too many images can also cause the distance calculations on the k-means
	 * algorithm to overflow. Recomended values are below 500.
	 */
	unsigned int textonImages;
	
	/**
	 * @brief Number of codewords to use in the Codebook.
	 * 
	 * This corresponds to the number of clusters to be calculated by a k-means
	 * clustering step.
	 */
	unsigned int codewords;
	
	/**
	 * @brief The type of histogram to use.
	 * 
	 * @pre Requires @a pyramidLevels to be higher than 0 to have any effect.
	 * @see Codebook::Type
	 * @see Settings::pyramidLevels
	 */
	Codebook::Type histogramType;
	
	/**
	 * @brief Number of levels to use when generating spatial pyramids.
	 *
	 * @warning The length of the histogram grows exponentially with the number
	 * of levels. Values below 3 are recommended.
	 */
	unsigned int pyramidLevels;
	
	/**
	 * @brief The C-SVM penalty value.
	 *
	 * Must be greater than 0. Values closer to 0 tend to underfit the data,
	 * while larger values tend to overfit.
	 */
	double C;
	
	/**
	 * @brief Number of images to use to train each class.
	 *
	 * @warning Must be lower than or equal to the number of images of
	 * the smallest class.
	 */
	unsigned int trainImagesPerClass;
};

#endif
