#ifndef CODEBOOK_H
#define CODEBOOK_H

extern "C" {
	#include <vl/kmeans.h>
}

#include <vector>

#include <boost/serialization/vector.hpp>

#include "features/ImageFeatures.h"
#include "codebook/Histogram.h"

/**
 * @brief Contains the codebook used to encode features.
 *
 * Each codeword represents the center of a k-means cluster. A feature is
 * encoded by determining the index of the codeword with the smallest Euclidean
 * distance to that feature.
 */
class Codebook {
public:
	/**
	 * @brief The type of sections to use when grouping features into an
	 * histogram
	 */
	enum Type {
		SQUARES, /**< will use a regular grid over the image */
		SLICES   /**< will split the image into several horizontal
				 	and vertical strips */
	};

	/**
	 * @brief Initializes a codebook.
	 *
	 * Sets up all the data required to encode new images into an histogram.
	 *
	 * @param clusterCenters An array of cluster centers,
	 * previously obtained using k-means.
	 * @param numClusters The number of cluster centers.
	 * @param dataSize The length of each descriptor.
	 * @param type The type of histogram division to use.
	 */
	Codebook(const float* clusterCenters, unsigned int numClusters,
		unsigned int dataSize, Type type);
	~Codebook();
	
	/**
	 * @brief Computes the histogram of one image.
	 *
	 * Maps image features to the previously calculated cluster centers to
	 * reduce the ammount of data and then encodes those new features into a
	 * spatial histogram.
	 *
	 * @param imageFeatures The features of one image which will be encoded
	 * using the codebook and then grouped into an histogram.
	 * @param levels The number of levels of the spatial pyramid to be built.
	 * @return The histogram which encodes the given image features.
	 */
	Histogram* computeHistogram(ImageFeatures* imageFeatures,
		unsigned int levels);
	
private:
	Type m_type;
	VlKMeans* m_kmeans;
	std::vector<float> m_centers;
	unsigned int m_numClusters;
	
	unsigned int histogramIndex(unsigned int level,
		unsigned int cellX, unsigned int cellY, unsigned int index) const;
	
	// Boost serialization
	friend class boost::serialization::access;
	Codebook();
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_type;
		ar & m_numClusters;
		ar & m_centers;
	}
};

#endif
