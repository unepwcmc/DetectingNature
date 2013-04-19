#ifndef FISHER_CODEBOOK_H
#define FISHER_CODEBOOK_H

extern "C" {
	#include <stdio.h>
	#include <yael/gmm.h>
	#include <yael/vector.h>
	#include <yael/matrix.h>
}

#include <vector>

#include <gmm.h>
#include <fisher.h>

#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>

#include "codebook/Codebook.h"
#include "features/ImageFeatures.h"
#include "codebook/Histogram.h"

/**
 * @brief Contains the codebook used to encode features into Fisher Vectors.
 *
 * The dimensionality of each feature is reduced using Principal Component
 * Analysis and is then encoded using soft-assignment to clusters and using the
 * distance, mean and variance as statistics.
 */
class FisherCodebook : public Codebook {
public:
	/**
	 * @brief Initializes a codebook.
	 *
	 * Sets up all the data required to encode new images into an histogram.
	 *
	 * @param gmm The Gaussian Mixture Model to be used to encode the features.
	 * @param pca Principal Component Analysis matrix used to reduce
	 * feature dimensionality.
	 */
	FisherCodebook(gaussian_mixture<float>* gmm,
		pca_online_t* pca, unsigned int pcaDim);
	~FisherCodebook();
	
	Histogram* encode(ImageFeatures* imageFeatures);
	
private:
	unsigned int m_pcaDim;
	fisher<float>* m_codebook;
	gaussian_mixture<float>* m_gmm;
	pca_online_t* m_pca;
	
	// Boost serialization
	friend class boost::serialization::access;
	FisherCodebook();
	BOOST_SERIALIZATION_SPLIT_MEMBER();
	template<class Archive>
	void save(Archive& ar, const unsigned int version) const
	{
		ar & boost::serialization::base_object<Codebook>(*this);
		
		m_gmm->save("fishercodebook");
		
		ar << m_pca->d;
		for(int i = 0; i < m_pca->d; i++) {
			ar << m_pca->mu[i];
		}
		for(int i = 0; i < m_pca->d * m_pca->d; i++) {
			ar << m_pca->eigvec[i];
		}
	}
	template<class Archive>
	void load(Archive& ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<Codebook>(*this);
		
		m_gmm = new gaussian_mixture<float>("fishercodebook");

		int d;
		ar >> d;
		m_pca = pca_online_new(d);
		for(int i = 0; i < m_pca->d; i++) {
			ar >> m_pca->mu[i];
		}
		for(int i = 0; i < m_pca->d * m_pca->d; i++) {
			ar >> m_pca->eigvec[i];
		}
	}
};

#endif
