#ifndef FEATURE_TRANSFORM_H
#define FEATURE_TRANSFORM_H

#include "features/ImageFeatures.h"

/**
 * @brief Transforms the features of an image.
 *
 * These transformations are applied on a set of previously extracted features
 * and can be chained together. The transformed features are not guaranteed to
 * be of the same dimension as the input data.
 */
class FeatureTransform {
public:
	virtual ~FeatureTransform() {};

	/**
	 * @brief Transform the features of one image.
	 *
	 * @warning The original image features will be deleted. Use the returned
	 * pointer instead.
	 *
	 * @param orig The original features to be processed.
	 * @return The result of applying the transformation to the
	 * original features.
	 */
	virtual ImageFeatures* transform(const ImageFeatures* orig) const = 0;
};

#endif
