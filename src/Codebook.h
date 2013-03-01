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
	Codebook(VlKMeans* kmeans, int numClusters);
	~Codebook();
	
	Histogram* computeHistogram(ImageFeatures* imageFeatures);
	
private:
	VlKMeans* m_kmeans;
	int m_numClusters;
};

#endif
