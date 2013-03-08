#ifndef CODEBOOK_H
#define CODEBOOK_H

extern "C" {
	#include <vl/kmeans.h>
}

#include <vector>

#include <boost/serialization/vector.hpp>

#include "ImageFeatures.h"
#include "Histogram.h"

class Codebook {
public:
	Codebook();
	Codebook(const float* clusterCenters,
		unsigned int numClusters, unsigned int dataSize);
	~Codebook();
	
	Histogram* computeHistogram(ImageFeatures* imageFeatures,
		unsigned int levels);
	
private:
	VlKMeans* m_kmeans;
	std::vector<float> m_centers;
	unsigned int m_numClusters;
	
	unsigned int histogramIndex(unsigned int level,
		unsigned int cellX, unsigned int cellY, unsigned int index);
	
	// Boost serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_numClusters;
		ar & m_centers;
	}
};

#endif
