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
	enum Type {DSIFT, HOG};

	FeatureExtractor(Type type, unsigned int gridSpacing,
		unsigned int patchSize);
	ImageFeatures* extract(Image& img);
	
private:
	Type m_type;
	unsigned int m_gridSpacing;
	unsigned int m_patchSize;

	ImageFeatures* extractDsift(Image& img);
	ImageFeatures* extractHog(Image& img);
};

#endif
