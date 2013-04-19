#include "FisherCodebookGenerator.h"
using namespace std;

FisherCodebookGenerator::FisherCodebookGenerator(
		const SettingsManager* settings) : CodebookGenerator(settings) {
	
	m_numClusters = settings->get<unsigned int>("codebook.codewords");
}

Codebook* FisherCodebookGenerator::generate(
		vector<ImageFeatures*> imageFeatures) const {
	
	vector<float> descriptors =	generateDescriptorSet(imageFeatures);
	unsigned int descriptorSize = imageFeatures[0]->getDescriptorSize();
	unsigned int numFeatures = descriptors.size() / descriptorSize;
	
	// Perform PCA
	unsigned int newDim = 80;
	pca_online_t* pca = pca_online_new(descriptorSize);
	pca_online_accu(pca, &descriptors[0], numFeatures);
	pca_online_complete(pca);
	
	vector<float> pcaFeatures(numFeatures * newDim, 0.0);
	pca_online_project(pca, &descriptors[0], &pcaFeatures[0],
		descriptorSize, numFeatures, newDim);
	descriptors.clear();

	// Spend some precious CPU cycles converting data into
	// the format expected by gmm-fisher
	vector<float*> samples(numFeatures, nullptr);
	for(unsigned int i = 0; i < numFeatures; i++) {
		samples[i] = new float[newDim];
		for(unsigned int j = 0; j < newDim; j++) {
			samples[i][j] = pcaFeatures[i * newDim + j];
		}
	}
	
	// Use k-means to initialize GMM
	float* distances = new float[numFeatures];
	float* centroids = new float[numFeatures * newDim];
	kmeans(newDim, numFeatures, m_numClusters, 1000, &pcaFeatures[0],
		4, -1, 1, centroids, distances, nullptr, nullptr);
	
	vector<float*> mean(m_numClusters, nullptr);
	for(unsigned int i = 0; i < m_numClusters; i++) {
		mean[i] = new float[newDim];
		for(unsigned int j = 0; j < newDim; j++) {
			mean[i][j] = centroids[i * newDim + j];
		}
	}
	delete[] centroids;
	
	vector<float*> var(m_numClusters, nullptr);
	float sigma = fvec_sum(distances, numFeatures) / numFeatures;
	delete[] distances;
	for(unsigned int i = 0; i < m_numClusters; i++) {
		var[i] = new float[newDim];
		for(unsigned int j = 0; j < newDim; j++) {
			var[i][j] = sigma;
		}
	}

	vector<float> coef(m_numClusters, 1.0 / m_numClusters);
	
	pcaFeatures.clear();
	
	// Calculate Gaussian Mixture Model
	gaussian_mixture<float>* gmm =
		new gaussian_mixture<float>(m_numClusters, newDim);
	//gmm->random_init(samples);
	gmm->set(mean, var, coef);
	for(unsigned int i = 0; i < m_numClusters; i++) {
		delete[] mean[i];
		delete[] var[i];
	}
	
	gmm->em(samples);
	for(unsigned int i = 0; i < numFeatures; i++) {
		delete[] samples[i];
	}
	
	return new FisherCodebook(gmm, pca);
}
