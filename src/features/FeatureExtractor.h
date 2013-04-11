#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

extern "C" {
	#include <vl/imopv.h>
	#include <vl/dsift.h>
	#include <vl/hog.h>
}

#include <bitset>

#include "images/ImageData.h"
#include "features/ImageFeatures.h"

/**
 * @brief Extracts the features of an image.
 *
 * These features are represented using descriptors, which usually represent an
 * image overview, invariant to scale, rotation and/or illumination.
 * As such they become useful when trying to compare different images of
 * the same scene.
 */
class FeatureExtractor {
public:
	virtual ~FeatureExtractor() {};
	/**
	 * @brief Extract the features of one image.
	 *
	 * If the image contains more than one channel, features will be extracted
	 * for each one of these and added to the ImageFeatures instance.
	 *
	 * @param img The raw image data to be processed.
	 * @return All the features extracted for this image.
	 */
	virtual ImageFeatures* extract(const ImageData* img) const = 0;
};

#endif
