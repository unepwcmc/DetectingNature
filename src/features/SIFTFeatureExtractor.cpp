#include "SIFTFeatureExtractor.h"
using namespace std;

SIFTFeatureExtractor::SIFTFeatureExtractor(const SettingsManager* settings) {
	m_smoothingSigma = settings->get<float>("image.smoothingSigma");
	m_gridSpacings = settings->get<vector<int> >("features.gridSpacing");
	m_patchSizes = settings->get<vector<int> >("features.patchSize");
}

ImageFeatures* SIFTFeatureExtractor::extract(const ImageData* img) const {
	ImageFeatures* imageFeatures = new ImageFeatures(
		img->getWidth(), img->getHeight(), img->getNumChannels());
	#pragma omp critical
	for(unsigned int h = 0; h < m_gridSpacings.size(); h++) {
		unsigned int binSize = m_patchSizes[h] / 4.0;
		unsigned int gridSpacing = m_gridSpacings[h];
	
		VlDsiftFilter* filter =
			vl_dsift_new_basic(img->getWidth(), img->getHeight(),
				gridSpacing, binSize);
		vl_dsift_set_flat_window(filter, true);
			
		for(unsigned int i = 0; i < img->getNumChannels(); i++) {
			if(m_smoothingSigma > 0.0) {
				float smoothedData[img->getHeight() * img->getWidth()];
				vl_imsmooth_f(smoothedData, img->getWidth(), img->getData(i),
					img->getWidth(), img->getHeight(), img->getWidth(),
					m_smoothingSigma, m_smoothingSigma);
				vl_dsift_process(filter, smoothedData);
			} else {
				vl_dsift_process(filter, img->getData(i));
			}
	
			unsigned int descriptorSize = vl_dsift_get_descriptor_size(filter);
			unsigned int numDescriptors = vl_dsift_get_keypoint_num(filter);
			float const* descriptors = vl_dsift_get_descriptors(filter);

			if(i == 0) {
				vector<pair<int, int> > coordinates;
				const VlDsiftKeypoint* keypoints = vl_dsift_get_keypoints(filter);
				for(unsigned int j = 0; j < numDescriptors; j++) {
					coordinates.push_back(
						make_pair(keypoints[j].x, keypoints[j].y));
				}
			
				imageFeatures->newFeatures(descriptors, descriptorSize,
					numDescriptors, coordinates);
			} else {
				imageFeatures->extendFeatures(i, descriptors, numDescriptors);
			}
		}
	
		vl_dsift_delete(filter);
	}
	return imageFeatures;
}
