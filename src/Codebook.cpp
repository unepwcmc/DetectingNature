#include "Codebook.h"
using namespace std;

Codebook::Codebook() {
	m_numClusters = 0;
}

Codebook::Codebook(const float* clusterCenters,
		unsigned int numClusters, unsigned int dataSize) {
	
	m_kmeans = nullptr;
	unsigned int totalSize = numClusters * dataSize;
	m_centers.reserve(totalSize);
	copy(clusterCenters, clusterCenters + totalSize, back_inserter(m_centers));
	m_numClusters = numClusters;
}

Codebook::~Codebook() {
	if(m_kmeans != nullptr) {
		vl_kmeans_delete(m_kmeans);
	}
}

unsigned int Codebook::histogramIndex(unsigned int level,
	unsigned int cellX, unsigned int cellY, unsigned int index) {
	
	unsigned int numDivisions = pow(2, level);
	
	unsigned int levelIndex = m_numClusters * (pow(4, level) - 1) / 3;
	unsigned int cellIndex = (cellX * numDivisions + cellY) * m_numClusters;
	
	return levelIndex + cellIndex + index;
}

Histogram* Codebook::computeHistogram(ImageFeatures* imageFeatures,
		unsigned int levels) {
	
	unsigned int totalLength = m_numClusters * (pow(4, levels + 1) - 1) / 3;
	vector<double> histogram(totalLength, 0);
	
	int numFeatures = imageFeatures->getNumFeatures();
	
	vl_uint32* assignments = new vl_uint32[numFeatures];
	vl_kmeans_quantize(m_kmeans, assignments, nullptr,
		imageFeatures->getFeatures(), numFeatures);
			
	for(int i = 0; i < numFeatures; i++) {
		pair<int, int> position = imageFeatures->getCoordinates(i);
		
		int numDivisions = pow(2, levels);
		
		unsigned int currentCellX = position.first /
			(float)imageFeatures->getWidth() * numDivisions;
		unsigned int currentCellY = position.second /
			(float)imageFeatures->getHeight() * numDivisions;

		histogram[histogramIndex(
			levels, currentCellX, currentCellY, assignments[i])]++;
	}
	delete[] assignments;
	
	for(int l = levels; l >= 0; l--) {
		unsigned int numDivisions = pow(2, l);
		for(unsigned int i = 0; i < numDivisions; i++) {
			for(unsigned int j = 0; j < numDivisions; j++) {			
				for(unsigned int k = 0; k < m_numClusters; k++) {
					if((unsigned int)l == levels) {
						histogram[histogramIndex(l, i, j, k)] /= numFeatures;
					} else {
						histogram[histogramIndex(l, i, j, k)] =
							histogram[histogramIndex(l+1, i*2, j*2, k)] +
							histogram[histogramIndex(l+1, i*2+1, j*2, k)] +
							histogram[histogramIndex(l+1, i*2, j*2+1, k)] +
							histogram[histogramIndex(l+1, i*2+1, j*2+1, k)];
					}
				}
			}
		}
	}
	
	for(int l = levels; l >= 0; l--) {	
		double levelWeight = (l == 0) ?
			1.0 / pow(2, levels) :
			1.0 / pow(2, levels - l + 1);
		
		unsigned int numDivisions = pow(2, l);
		for(unsigned int i = 0; i < numDivisions; i++) {
			for(unsigned int j = 0; j < numDivisions; j++) {			
				for(unsigned int k = 0; k < m_numClusters; k++) {
					histogram[histogramIndex(l, i, j, k)] *= levelWeight;
				}
			}
		}
	}
	
	return new Histogram(&histogram[0], totalLength);
}
