#ifndef FISHER_CODEBOOK_GENERATOR_H
#define FISHER_CODEBOOK_GENERATOR_H

extern "C" {
	#include <stdio.h>
	#include <yael/kmeans.h>
	#include <yael/matrix.h>
	#include <yael/vector.h>
}

#include <vector>
#include <random>
#include <algorithm>

#include <gmm.h>
#include <fisher.h>

#include "framework/SettingsManager.h"
#include "utils/OutputHelper.h"
#include "features/ImageFeatures.h"
#include "codebook/CodebookGenerator.h"
#include "codebook/FisherCodebook.h"

/**
 * @brief Creates a codebook capable of creating Fisher Vectors.
 *
 * The codebook attempts to reduce the dimensionality of the data by clustering
 * similar features into a single codeword. This is done by generating a
 * Gaussian Mixture Model and using soft-assignment to encode features.
 */
class FisherCodebookGenerator : public CodebookGenerator {
public:
	/**
	 * @brief Initializes the codebook generator with the image data.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	FisherCodebookGenerator(const SettingsManager* settings);
	
	Codebook* generate(std::vector<ImageFeatures*> imageFeatures) const;
	
private:
	unsigned int m_pcaDim;
	unsigned int m_numClusters;
};

#endif
