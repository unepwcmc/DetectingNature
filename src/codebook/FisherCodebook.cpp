#include "FisherCodebook.h"
using namespace std;

FisherCodebook::FisherCodebook() {
	m_codebook = nullptr;
}

FisherCodebook::FisherCodebook(gaussian_mixture<float>* gmm,
		pca_online_t* pca, unsigned int pcaDim) {
	
	m_pcaDim = pcaDim;
	m_gmm = gmm;
	m_codebook = nullptr;
	m_pca = pca;
}

FisherCodebook::~FisherCodebook() {
	delete m_gmm;
	pca_online_delete(m_pca);
		
	if(m_codebook != nullptr) {
		delete m_codebook;
	}
}

Histogram* FisherCodebook::encode(ImageFeatures* imageFeatures) {
	#pragma omp critical
	if(m_codebook == nullptr) {
		fisher_param params;
		params.grad_weights = true;
		params.grad_means = true;
    	params.grad_variances = true;
		m_codebook = new fisher<float>(params);
		m_codebook->set_model(*m_gmm);
	}
	
	unsigned int numFeatures = imageFeatures->getNumFeatures();

	vector<float> pcaFeatures(numFeatures * m_pcaDim, 0.0);
	pca_online_project(m_pca, imageFeatures->getFeatures(), &pcaFeatures[0],
		imageFeatures->getDescriptorSize(), numFeatures, m_pcaDim);

	vector<float*> samples(numFeatures, nullptr);
	for(unsigned int i = 0; i < numFeatures; i++) {
		samples[i] = new float[m_pcaDim];
		for(unsigned int j = 0; j < m_pcaDim; j++) {
			samples[i][j] = pcaFeatures[i * m_pcaDim + j];
		}
	}

	unsigned int fisherLength = m_codebook->dim();
	float result[fisherLength];
	m_codebook->compute(samples, result);
	
	for(unsigned int i = 0; i < numFeatures; i++) {
		delete[] samples[i];
	}
	
	vector<double> histogram(result, result + fisherLength);	
	return new Histogram(&histogram[0], fisherLength);
}
