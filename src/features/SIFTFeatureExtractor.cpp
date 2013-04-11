#include "SIFTFeatureExtractor.h"
using namespace std;

SIFTFeatureExtractor::SIFTFeatureExtractor(const SettingsManager* settings) {
	m_smoothingSigma = settings->get<float>("image.smoothingSigma");
	m_gridSpacing = settings->get<int>("features.gridSpacing");
	m_patchSize = settings->get<int>("features.patchSize");
}

ImageFeatures* SIFTFeatureExtractor::extract(Image& img) const {
	ImageFeatures* imageFeatures = new ImageFeatures(
		img.getWidth(), img.getHeight(), img.getNumChannels());
	
	unsigned int numConcentricDesc = 8;
	unsigned int mainBinSize = m_patchSize / 4;
	for(unsigned int h = 0; h < numConcentricDesc; h++) {
		unsigned int binSize = mainBinSize + 1.2 * h;
	
		VlDsiftFilter* filter =
			vl_dsift_new_basic(img.getWidth(), img.getHeight(),
				binSize * 2, binSize);
		int margin = (binSize / 2);// + (3.0 / 2.0 * (mainBinSize - binSize));
		vl_dsift_set_bounds(filter, margin, margin,
			img.getWidth() - margin - 1, img.getHeight() - margin - 1);
		vl_dsift_set_flat_window(filter, true);
	
		for(unsigned int i = 0; i < img.getNumChannels(); i++) {
			if(m_smoothingSigma > 0.0) {
				float smoothedData[img.getHeight() * img.getWidth()];
				vl_imsmooth_f(smoothedData, img.getWidth(), img.getData(i),
					img.getWidth(), img.getHeight(), img.getWidth(),
					m_smoothingSigma, m_smoothingSigma);
				vl_dsift_process(filter, smoothedData);
			} else {
				vl_dsift_process(filter, img.getData(i));
			}
	
			unsigned int descriptorSize = vl_dsift_get_descriptor_size(filter);
			unsigned int numDescriptors = vl_dsift_get_keypoint_num(filter);
			float const* descriptors = vl_dsift_get_descriptors(filter);

			vector<pair<int, int> > coordinates;
			const VlDsiftKeypoint* keypoints = vl_dsift_get_keypoints(filter);
			for(unsigned int j = 0; j < numDescriptors; j++) {
				coordinates.push_back(
					make_pair(keypoints[j].x, keypoints[j].y));
			}

			if(i == 0) {
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
