#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

extern "C" {
	#include <vl/imopv.h>
	#include <vl/dsift.h>
	#include <vl/hog.h>
}

#include <bitset>

#include "features/Image.h"
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
	/**
	 * @brief The type of descriptor to use to represent the image features
	 */
	enum Type {
		DSIFT, /**< Uses dense Scale Invariant Feature Transform descriptors */
		HOG, /**< Uses Histograms of Oriented Gradients descriptors */
		LBP /**< Used Local Binary Pattern descriptors */
	};

	/**
	 * @brief Sets up the feature extraction parameters.
	 *
	 * @param type Selects the type of descriptor to use.
	 * @param smoothingSigma How much smoothing to apply to the image before
	 * extracting the features. Zero disables it.
	 * @param gridSpacing The distance between each keypoint on a dense grid.
	 * @param patchSize The size of the descriptor. Values larger than
	 * @a gridSpacing result in overlapping descriptors which usually have
	 * better results when classifying images.
	 */
	FeatureExtractor(Type type, float smoothingSigma,
		unsigned int gridSpacing, unsigned int patchSize);
	
	/**
	 * @brief Extract the features of one image.
	 *
	 * If the image contains more than one channel, features will be extracted
	 * for each one of these and added to the ImageFeatures instance.
	 *
	 * @param img The raw image data to be processed.
	 * @return All the features extracted for this image.
	 */
	ImageFeatures* extract(Image& img) const;
	
private:
	Type m_type;
	float m_smoothingSigma;
	unsigned int m_gridSpacing;
	unsigned int m_patchSize;

	ImageFeatures* extractDsift(Image& img) const;
	ImageFeatures* extractHog(Image& img) const;
	ImageFeatures* extractLbp(Image& img) const;
	float* stackFeatures(float* descriptors, unsigned int descriptorSize,
		unsigned int numDescriptors, unsigned int width,
		unsigned int height, unsigned int numStacks) const;
};

#endif
