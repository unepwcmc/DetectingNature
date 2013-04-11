#ifndef HELLINGER_FEATURE_TRANSFORM_H
#define HELLINGER_FEATURE_TRANSFORM_H

#include "framework/SettingsManager.h"
#include "features/FeatureTransform.h"
#include "features/ImageFeatures.h"

/**
 * @brief Applies an approximation of the Hellinger kernel to the features.
 *
 * By applying this tranformation to a set of features, calculating the
 * Euclidean distance between any feature pair will result, approximately,
 * in the same value as the one of the Hellinger kernel for that pair.
 */
class HellingerFeatureTransform : public FeatureTransform {
public:
	/**
	 * @brief Initializes an instance of the Hellinger transform.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	HellingerFeatureTransform(const SettingsManager* settings);
	ImageFeatures* transform(const ImageFeatures* orig) const;
};

#endif
