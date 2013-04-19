#ifndef KMEANS_CODEBOOK_GENERATOR_H
#define KMEANS_CODEBOOK_GENERATOR_H

extern "C" {
	#include <vl/kmeans.h>
}

#include <vector>
#include <random>
#include <algorithm>

#include "framework/SettingsManager.h"
#include "codebook/CodebookGenerator.h"
#include "utils/OutputHelper.h"
#include "features/ImageFeatures.h"
#include "codebook/KMeansCodebook.h"

/**
 * @brief Creates a codebook capable of creating Spatial Pyramids.
 *
 * The codebook attempts to reduce the dimensionality of the data by clustering
 * similar features into a single codeword. This is done by using a clustering
 * algorithm, such as k-means.
 */
class KMeansCodebookGenerator : public CodebookGenerator {
public:
	/**
	 * @brief Initializes the codebook generator with the image data.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 */
	KMeansCodebookGenerator(const SettingsManager* settings);
	
	Codebook* generate(std::vector<ImageFeatures*> imageFeatures) const;
	
private:
	unsigned int m_numClusters;
	unsigned int m_levels;
	KMeansCodebook::Type m_type;
};

#endif
