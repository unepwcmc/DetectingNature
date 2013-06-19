#ifndef CODEBOOK_GENERATOR_H
#define CODEBOOK_GENERATOR_H

#include <vector>

#include "framework/SettingsManager.h"
#include "features/ImageFeatures.h"
#include "codebook/Codebook.h"

/**
 * @brief Creates a codebook based on some representative image features.
 *
 * The codebook attempts to reduce the dimensionality of the data by clustering
 * similar features into a single codeword.
 */
class CodebookGenerator {
public:
	CodebookGenerator(const SettingsManager* settings);

	/**
	 * @brief Generates a codebook using the set of image features defined in
	 * constructor.
	 *
	 * The resulting codebook will be capable of encoding an image into an
	 * histogram.
	 *
	 * @warning This is a memory intensive process since a large number of
	 * image features must be kept in memory. Once the codebook has been
	 * generated these features can be deleted.
	 *
	 * @param imageFeatures A vector containing the image features to be used
	 * in the encoding process.
	 * @return A codebook capable of encoding new images into an histogram.
	 */
	virtual Codebook* generate(
		std::vector<ImageFeatures*> imageFeatures) const = 0;

protected:
	unsigned int m_numFeatures;
	
	std::vector<float> generateDescriptorSet(
		std::vector<ImageFeatures*> imageFeatures) const;
};

#endif
