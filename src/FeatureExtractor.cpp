#include "FeatureExtractor.h"
using namespace std;

FeatureExtractor::FeatureExtractor(Type type,
		unsigned int gridSpacing, unsigned int patchSize) {
	
	m_type = type;
	m_gridSpacing = gridSpacing;
	m_patchSize = patchSize;
}

ImageFeatures* FeatureExtractor::extract(Image& img) {
	switch(m_type) {
	case HOG:
		return extractHog(img);
	case DSIFT:
		return extractDsift(img);
	default:
		return nullptr;
	}
}

ImageFeatures* FeatureExtractor::extractDsift(Image& img) {
	VlDsiftFilter* filter =
		vl_dsift_new_basic(img.getWidth(), img.getHeight(),
			m_gridSpacing, m_patchSize / 4);
	
	ImageFeatures* imageFeatures =
		new ImageFeatures(
			img.getWidth(), img.getHeight(), img.getNumChannels());
	
	for(unsigned int i = 0; i < img.getNumChannels(); i++) {
		vl_dsift_process(filter, img.getData(i));
		
		unsigned int descriptorSize = vl_dsift_get_descriptor_size(filter);
		unsigned int numDescriptors = vl_dsift_get_keypoint_num(filter);
		float const* descriptors = vl_dsift_get_descriptors(filter);
	
		vector<pair<int, int> > coordinates;
		const VlDsiftKeypoint* keypoints = vl_dsift_get_keypoints(filter);
		for(unsigned int j = 0; j < numDescriptors; j++) {
			coordinates.push_back(make_pair(keypoints[j].x, keypoints[j].y));
		}
	
		imageFeatures->addFeatures(i, descriptors, descriptorSize,
			numDescriptors, coordinates);
	}
	
	vl_dsift_delete(filter);
	
	return imageFeatures;
}

ImageFeatures* FeatureExtractor::extractHog(Image& img) {
/*
	VlHog* hog = vl_hog_new(VlHogVariantDalalTriggs, 8, false);
	vl_hog_put_image(hog, img.getData(), img.getWidth(), img.getHeight(),
		1, 8);
		
	int descriptorSize = vl_hog_get_dimension(hog);
	int numDescriptors = vl_hog_get_width(hog) * vl_hog_get_height(hog);
	
	float* descriptors =
		new float[descriptorSize * numDescriptors];
	vl_hog_extract(hog, descriptors);
	
	ImageFeatures* imageFeatures =
		new ImageFeatures(descriptors, descriptorSize, numDescriptors);
	
	delete[] descriptors;
	vl_hog_delete(hog);
	
	return imageFeatures;
*/
	return nullptr;
}
