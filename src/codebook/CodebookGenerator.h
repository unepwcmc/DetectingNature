#ifndef CODEBOOK_GENERATOR_H
#define CODEBOOK_GENERATOR_H

extern "C" {
	#include <vl/kmeans.h>
}

#include <vector>
#include <random>
#include <algorithm>

#include "utils/OutputHelper.h"
#include "features/ImageFeatures.h"
#include "codebook/Codebook.h"

/**
 * @brief Creates a codebook based on some representative image features.
 *
 * The codebook attempts to reduce the dimensionality of the data by clustering
 * similar features into a single codeword. This is done by using a clustering
 * algorithm, such as k-means.
 */
class CodebookGenerator {
public:
	/**
	 * @brief Initializes the codebook generator with the image data.
	 *
	 * @param imageFeatures A vector containing the image features to be used
	 * in the clustering process.
	 */
	CodebookGenerator(std::vector<ImageFeatures*> imageFeatures);
	
	/**
	 * @brief Generates a codebook using the set of image features defined in
	 * constructor.
	 *
	 * The resulting codebook will be capable of encoding an image into an
	 * histogram of @a numClusters length.
	 *
	 * @warning This is a memory intensive process since a large number of
	 * image features must be kept in memory. Once the clustering is done these
	 * features can be deleted.
	 *
	 * @param numTextonImages The number or images to use during the clustering
	 * process.
	 * @param numClusters How many clusters to generate using the given image
	 * features.
	 * @param type The type of histogram the resulting codebook will generate.
	 * @return A codebook capable of encoding new images into an histogram.
	 */
	Codebook* generate(unsigned int numTextonImages,
		unsigned int numClusters, Codebook::Type type) const;
	
private:
	std::vector<ImageFeatures*> m_imageFeatures;
	
	std::vector<float> generateDescriptorSet(
		unsigned int numTextonImages) const;
};

#endif
