#ifndef CODEBOOK_GENERATOR_H
#define CODEBOOK_GENERATOR_H

extern "C" {
	#include <vl/kmeans.h>
}

#include <vector>
#include <random>
#include <algorithm>

#include "OutputHelper.h"
#include "ImageFeatures.h"
#include "Codebook.h"

class CodebookGenerator {
public:
	CodebookGenerator(std::vector<ImageFeatures*> imageFeatures);
	Codebook* generate(unsigned int numTextonImages, unsigned int numClusters);
	
private:
	std::vector<ImageFeatures*> m_imageFeatures;
	
	std::vector<float> generateDescriptorSet(unsigned int numTextonImages);
};

#endif
