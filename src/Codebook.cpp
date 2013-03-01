#include "Codebook.h"
using namespace std;

Codebook::Codebook(VlKMeans* kmeans, int numClusters) {
	m_kmeans = kmeans;
	m_numClusters = numClusters;
}

Codebook::~Codebook() {
	vl_kmeans_delete(m_kmeans);
}

Histogram* Codebook::computeHistogram(ImageFeatures* imageFeatures) {
	int numFeatures = imageFeatures->getNumFeatures();
	vector<float> histogram(m_numClusters, 0);
	
	vl_uint32* assignments = new vl_uint32[numFeatures];
	vl_kmeans_quantize(m_kmeans, assignments, nullptr,
		imageFeatures->getFeatures(), numFeatures);
				
	for(int i = 0; i < numFeatures; i++) {
		histogram[assignments[i]]++;
	}
	delete[] assignments;
	
	for(int i = 0; i < m_numClusters; i++) {
		histogram[i] /= numFeatures;
	}
	
	return new Histogram(&histogram[0], m_numClusters);
}
