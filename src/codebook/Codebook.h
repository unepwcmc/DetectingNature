#ifndef CODEBOOK_H
#define CODEBOOK_H

#include "features/ImageFeatures.h"
#include "codebook/Histogram.h"

/**
 * @brief Represents a codebook used to encode features.
 *
 * Each codeword represents the center of a k-means cluster. A feature is
 * encoded by determining the index of the codeword with the smallest Euclidean
 * distance to that feature.
 */
class Codebook {
public:
	virtual ~Codebook() {}

	/**
	 * @brief Computes the histogram of one image.
	 *
	 * Maps image features to the previously calculated cluster centers to
	 * reduce the ammount of data and then encodes those new features into a
	 * spatial histogram.
	 *
	 * @param imageFeatures The features of one image which will be encoded
	 * using the codebook and then grouped into an histogram.
	 * @return The histogram which encodes the given image features.
	 */
	virtual Histogram* encode(ImageFeatures* imageFeatures) = 0;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {}
};

#endif
