#ifndef CODEBOOK_H
#define CODEBOOK_H

extern "C" {
	#include <vl/kmeans.h>
}

#include <vector>

#include "ImageFeatures.h"
#include "Histogram.h"

class Codebook {
public:
	Codebook(VlKMeans* kmeans, unsigned int numClusters);
	~Codebook();
	
	Histogram* computeHistogram(ImageFeatures* imageFeatures,
		unsigned int levels);
	
private:
	VlKMeans* m_kmeans;
	unsigned int m_numClusters;
	
	unsigned int histogramIndex(unsigned int level,
		unsigned int cellX, unsigned int cellY, unsigned int index);
};

#endif
