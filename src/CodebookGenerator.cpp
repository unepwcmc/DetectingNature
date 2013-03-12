#include "CodebookGenerator.h"
using namespace std;

CodebookGenerator::CodebookGenerator(vector<ImageFeatures*> imageFeatures) {
	m_imageFeatures = imageFeatures;
}

vector<float> CodebookGenerator::generateDescriptorSet(
		unsigned int numTextonImages) {
		
	int descriptorSize = m_imageFeatures[0]->getDescriptorSize();
	vector<float> descriptors;
	
	for(unsigned int i = 0; i < numTextonImages; i++) {			
		for(unsigned int j = 0; j < m_imageFeatures[i]->getNumFeatures(); j++) {
			const float* feature = m_imageFeatures[i]->getFeature(j);
			copy(feature, feature + descriptorSize, back_inserter(descriptors));
		}
	}
	
	return descriptors;
}

Codebook* CodebookGenerator::generate(unsigned int numTextonImages,
		unsigned int numClusters, Codebook::Type type) {
		
	int descriptorSize = m_imageFeatures[0]->getDescriptorSize();
	vector<float> descriptors = generateDescriptorSet(numTextonImages);

	vl_set_printf_func(printVlfeat);

	VlKMeans* kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
	vl_kmeans_set_verbosity(kmeans, 1);
	vl_kmeans_set_initialization(kmeans, VlKMeansPlusPlus);
	vl_kmeans_set_algorithm(kmeans, VlKMeansElkan);
	vl_kmeans_set_num_repetitions(kmeans, 1);
	vl_kmeans_set_max_num_iterations(kmeans, 500);
	vl_kmeans_cluster(kmeans, &descriptors[0],
		descriptorSize, descriptors.size() / descriptorSize, numClusters);
	
	return new Codebook((const float*)vl_kmeans_get_centers(kmeans),
		numClusters, descriptorSize, type);
}
