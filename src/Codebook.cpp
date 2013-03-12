#include "Codebook.h"
using namespace std;

Codebook::Codebook() {
	m_kmeans = nullptr;
	m_numClusters = 0;
}

Codebook::Codebook(const float* clusterCenters, unsigned int numClusters,
		unsigned int dataSize, Type type) {
	
	m_type = type;
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
	
	unsigned int numDivisions = (m_type == SQUARES) ?
		pow(2, level) : pow(2, level + 1);
	
	unsigned int levelIndex = (m_type == SQUARES) ?
		m_numClusters * (pow(4, level) - 1) / 3 :
		m_numClusters * 4 * (pow(2, level) - 1);
	unsigned int cellIndex = (cellX * numDivisions + cellY) * m_numClusters;
	
	return levelIndex + cellIndex + index;
}

Histogram* Codebook::computeHistogram(ImageFeatures* imageFeatures,
		unsigned int levels) {
	
	if(m_kmeans == nullptr) {
		m_kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
		vl_kmeans_set_algorithm(m_kmeans, VlKMeansElkan);
		vl_kmeans_set_centers(m_kmeans, &m_centers[0],
			m_centers.size() / m_numClusters, m_numClusters);
	}
	
	unsigned int totalLength = (m_type == SQUARES) ?
		m_numClusters * (pow(4, levels + 1) - 1) / 3 :
		m_numClusters * 4 * (pow(2, levels + 1) - 1);
		
	vector<double> histogram(totalLength, 0);
	
	int numFeatures = imageFeatures->getNumFeatures();
	
	vl_uint32* assignments = new vl_uint32[numFeatures];
	vl_kmeans_quantize(m_kmeans, assignments, nullptr,
		imageFeatures->getFeatures(), numFeatures);
			
	for(int i = 0; i < numFeatures; i++) {
		pair<int, int> position = imageFeatures->getCoordinates(i);
		
		int numDivisions = (m_type == SQUARES) ?
			pow(2, levels) :
			pow(2, levels+1);
		
		unsigned int currentCellX = position.first /
			(float)imageFeatures->getWidth() * numDivisions;
		unsigned int currentCellY = position.second /
			(float)imageFeatures->getHeight() * numDivisions;

		if(m_type == SQUARES) {
			histogram[histogramIndex(
				levels, currentCellX, currentCellY, assignments[i])]++;
		} else {
			histogram[histogramIndex(
				levels, 0, currentCellX, assignments[i])]++;
			histogram[histogramIndex(
				levels, 1, currentCellY, assignments[i])]++;
		}
	}
	delete[] assignments;
	
	for(int l = levels; l >= 0; l--) {
		unsigned int numDivisionsX = (m_type == SQUARES) ?
			pow(2, l) : 2;
		unsigned int numDivisionsY = (m_type == SQUARES) ?
			pow(2, l) : pow(2, l+1);
			
		for(unsigned int i = 0; i < numDivisionsX; i++) {
			for(unsigned int j = 0; j < numDivisionsY; j++) {
				for(unsigned int k = 0; k < m_numClusters; k++) {
					if((unsigned int)l == levels) {
						histogram[histogramIndex(l, i, j, k)] /= numFeatures;
					} else {
						if(m_type == SQUARES) {
							histogram[histogramIndex(l, i, j, k)] =
								histogram[histogramIndex(l+1, i*2, j*2, k)] +
								histogram[histogramIndex(l+1, i*2+1, j*2, k)] +
								histogram[histogramIndex(l+1, i*2, j*2+1, k)] +
								histogram[histogramIndex(l+1, i*2+1, j*2+1, k)];
						} else {
							histogram[histogramIndex(l, i, j, k)] =
								histogram[histogramIndex(l+1, i, j*2, k)] +
								histogram[histogramIndex(l+1, i, j*2+1, k)];
						}
					}
				}
			}
		}
	}

	for(int l = levels; l >= 0; l--) {
		int lvl = (m_type == SQUARES) ?
			l : levels - l;
		double levelWeight = (lvl == 0) ?
			1.0 / pow(2, levels) :
			1.0 / pow(2, levels - lvl + 1);
		
		unsigned int numDivisionsX = (m_type == SQUARES) ?
			pow(2, l) : 2;
		unsigned int numDivisionsY = (m_type == SQUARES) ?
			pow(2, l) : pow(2, l+1);

		for(unsigned int i = 0; i < numDivisionsX; i++) {
			for(unsigned int j = 0; j < numDivisionsY; j++) {			
				for(unsigned int k = 0; k < m_numClusters; k++) {
					histogram[histogramIndex(l, i, j, k)] *= levelWeight;
				}
			}
		}
	}

	return new Histogram(&histogram[0], totalLength);
}
