#ifndef HOG_FEATURE_EXTRACTOR_H
#define HOG_FEATURE_EXTRACTOR_H

extern "C" {
	#include <vl/hog.h>
}

#include "features/FeatureExtractor.h"
#include "features/Image.h"
#include "features/ImageFeatures.h"
#include "framework/SettingsManager.h"

/**
 * @brief Extracts "Histogram of Oriented Gradients" descriptors from an image.
 *
 * HOGs are extracted on a regular grid and their main advantage is the
 * low computational cost required to extract them.
 */
class HOGFeatureExtractor : public FeatureExtractor {
public:
	/**
	 * @brief Sets up the feature extraction parameters.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	HOGFeatureExtractor(const SettingsManager* settings);
	
	ImageFeatures* extract(Image& img) const;
	
private:
	unsigned int m_gridSpacing;
	unsigned int m_patchSize;
	
	float* stackFeatures(float* descriptors,
		unsigned int descriptorSize, unsigned int numDescriptors,
		unsigned int width, unsigned int height, unsigned int numStacks) const;
};

#endif
