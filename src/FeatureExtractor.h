#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

extern "C" {
	#include <vl/dsift.h>
	#include <vl/hog.h>
}

#include "Image.h"
#include "ImageFeatures.h"

class FeatureExtractor {
public:
	FeatureExtractor();
	ImageFeatures* extractDsift(Image& img);
	ImageFeatures* extractHog(Image& img);
};

#endif
