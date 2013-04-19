#include "KMeansCodebookGenerator.h"
using namespace std;

KMeansCodebookGenerator::KMeansCodebookGenerator(
		const SettingsManager* settings) : CodebookGenerator(settings) {
		
	m_numClusters = settings->get<unsigned int>("codebook.codewords");
	m_levels = settings->get<unsigned int>("histogram.pyramidLevels");
	m_type = settings->get<string>("histogram.type") == "Slices" ?
		KMeansCodebook::SLICES : KMeansCodebook::SQUARES;
}

Codebook* KMeansCodebookGenerator::generate(
		vector<ImageFeatures*> imageFeatures) const {
		
	int descriptorSize = imageFeatures[0]->getDescriptorSize();
	vector<float> descriptors = generateDescriptorSet(imageFeatures);

	vl_set_printf_func(printVlfeat);

	VlKMeans* kmeans = vl_kmeans_new(VL_TYPE_FLOAT, VlDistanceL2);
	vl_kmeans_set_verbosity(kmeans, 1);
	vl_kmeans_set_initialization(kmeans, VlKMeansPlusPlus);
	vl_kmeans_set_algorithm(kmeans, VlKMeansElkan);
	vl_kmeans_set_num_repetitions(kmeans, 1);
	vl_kmeans_set_max_num_iterations(kmeans, 500);
	vl_kmeans_cluster(kmeans, &descriptors[0],
		descriptorSize, descriptors.size() / descriptorSize, m_numClusters);
	
	return new KMeansCodebook((const float*)vl_kmeans_get_centers(kmeans),
		m_numClusters, descriptorSize, m_type, m_levels);
}
