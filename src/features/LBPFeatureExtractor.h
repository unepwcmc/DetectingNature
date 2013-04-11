#ifndef LBP_FEATURE_EXTRACTOR_H
#define LBP_FEATURE_EXTRACTOR_H

#include <bitset>

#include "features/FeatureExtractor.h"
#include "framework/SettingsManager.h"

/**
 * @brief Extracts "Local Binary Pattern" descriptors from an image.
 *
 * These features are determined by calculating, for every pixel, a pattern
 * representing the gradient between that pixel and all its neighbours.
 * These descriptors are very compact and easy to calculate, while providing
 * results close to those of more complex descriptors.
 */
class LBPFeatureExtractor : public FeatureExtractor {
public:
	/**
	 * @brief Sets up the feature extraction parameters.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	LBPFeatureExtractor(const SettingsManager* settings);
	
	ImageFeatures* extract(const ImageData* img) const;
	
private:
	unsigned int m_gridSpacing;
	unsigned int m_patchSize;
};

#endif
