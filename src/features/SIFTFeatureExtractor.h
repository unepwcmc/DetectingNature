#ifndef SIFT_FEATURE_EXTRACTOR_H
#define SIFT_FEATURE_EXTRACTOR_H

extern "C" {
	#include <vl/imopv.h>
	#include <vl/dsift.h>
}

#include "features/FeatureExtractor.h"
#include "framework/SettingsManager.h"

/**
 * @brief Extracts dense "Scale Invariant Feature Transform" descriptors.
 *
 * These descriptors have a very high descriptive strength, allowing them
 * to achieve better classification results than other descriptors.
 */
class SIFTFeatureExtractor : public FeatureExtractor {
public:
	/**
	 * @brief Sets up the feature extraction parameters.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	SIFTFeatureExtractor(const SettingsManager* settings);

	ImageFeatures* extract(const ImageData* img) const;
	
private:
	float m_smoothingSigma;
	unsigned int m_gridSpacing;
	unsigned int m_patchSize;
};

#endif
