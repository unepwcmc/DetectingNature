#include "Codebook.h"
using namespace std;

Codebook::Codebook(VlKMeans* kmeans, unsigned int numClusters) {
	m_kmeans = kmeans;
	m_numClusters = numClusters;
}

Codebook::~Codebook() {
	vl_kmeans_delete(m_kmeans);
}

unsigned int Codebook::calculateHistogramIndex(unsigned int level,
	unsigned int cellX, unsigned int cellY) {
	
	unsigned int numDivisions = pow(2, level);
	
	unsigned int levelIndex = m_numClusters * (pow(4, level) - 1) / 3;
	unsigned int cellIndex = (cellX * numDivisions + cellY) * m_numClusters;
	
	return levelIndex + cellIndex;
}

Histogram* Codebook::computeHistogram(ImageFeatures* imageFeatures,
		unsigned int levels) {
	
	unsigned int totalLength = m_numClusters * (pow(4, levels + 1) - 1) / 3;
	vector<float> histogram(totalLength, 0);
	
	int numFeatures = imageFeatures->getNumFeatures();
	
	vl_uint32* assignments = new vl_uint32[numFeatures];
	vl_kmeans_quantize(m_kmeans, assignments, nullptr,
		imageFeatures->getFeatures(), numFeatures);
			
	for(int i = 0; i < numFeatures; i++) {
		pair<int, int> position = imageFeatures->getCoordinates(i);
		
		for(unsigned int l = 0; l <= levels; l++) {
			int numDivisions = pow(2, l);
			
			unsigned int currentCellX = position.first /
				(float)imageFeatures->getWidth() * numDivisions;
			unsigned int currentCellY = position.second /
				(float)imageFeatures->getHeight() * numDivisions;

			int histogramIndex = calculateHistogramIndex(
				l, currentCellX, currentCellY);

			histogram[histogramIndex + assignments[i]]++;
		}
	}
	delete[] assignments;

	
	for(unsigned int l = 0; l <= levels; l++) {
		unsigned int numDivisions = pow(2, l);
		float levelWeight = (l == 0) ?
			1.0 / pow(2, levels) :
			1.0 / pow(2, levels - l + 1);

		for(unsigned int i = 0; i < numDivisions; i++) {
			for(unsigned int j = 0; j < numDivisions; j++) {
				float histogramTotal = 0;
				unsigned int histogramIndex = calculateHistogramIndex(l, i, j);
				
				for(unsigned int k = 0; k < m_numClusters; k++) {
					histogramTotal += histogram[histogramIndex + k];
				}
				
				for(unsigned int k = 0; k < m_numClusters; k++) {
					histogram[histogramIndex + k] /= histogramTotal;
					histogram[histogramIndex + k] *= levelWeight;
				}
			}
		}
	}
	
	return new Histogram(&histogram[0], totalLength);
}
