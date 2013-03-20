#include "FeatureExtractor.h"
using namespace std;

FeatureExtractor::FeatureExtractor(Type type, float smoothingSigma,
		unsigned int gridSpacing, unsigned int patchSize) {
	
	m_smoothingSigma = smoothingSigma;
	m_type = type;
	m_gridSpacing = gridSpacing;
	m_patchSize = patchSize;
}

ImageFeatures* FeatureExtractor::extract(Image& img) const {
	switch(m_type) {
	case HOG:
		return extractHog(img);
	case DSIFT:
		return extractDsift(img);
	default:
		return nullptr;
	}
}

float* FeatureExtractor::stackFeatures(float* descriptors,
		unsigned int descriptorSize, unsigned int numDescriptors,
		unsigned int width, unsigned int height, unsigned int numStacks) const {
	
	unsigned int stride = descriptorSize * numStacks * numStacks;
	float* newDescriptors = new float[
		(width - numStacks + 1) * (height - numStacks + 1) * stride];
	
	float* newDescPtr = newDescriptors;
	float* descPtr = descriptors;
	for(unsigned int y = 0; y < (height - numStacks + 1); y++) {
		for(unsigned int x = 0; x < (width - numStacks + 1); x++) {	
			for(unsigned int dx = 0; dx < numStacks; dx++) {
				for(unsigned int dy = 0; dy < numStacks; dy++) {
					unsigned int offset =  ((x + dx) * descriptorSize) +
						((y + dy) * descriptorSize * width);
					copy(descPtr + offset,
						descPtr + offset + descriptorSize, newDescPtr);
					newDescPtr += descriptorSize;
				}
			}
		}
	}
	return newDescriptors;
}

ImageFeatures* FeatureExtractor::extractDsift(Image& img) const {
	VlDsiftFilter* filter =
		vl_dsift_new_basic(img.getWidth(), img.getHeight(),
			m_gridSpacing, m_patchSize / 4);
	int margin = m_patchSize / 8;
	vl_dsift_set_bounds(filter, margin, margin,
		img.getWidth() - margin - 1, img.getHeight() - margin - 1);
	vl_dsift_set_window_size(filter, 4);
	
	ImageFeatures* imageFeatures = new ImageFeatures(
			img.getWidth(), img.getHeight(), img.getNumChannels());
	
	for(unsigned int i = 0; i < img.getNumChannels(); i++) {
		float smoothedData[img.getHeight() * img.getWidth()];
		vl_imsmooth_f(smoothedData, img.getWidth(), img.getData(i),
			img.getWidth(), img.getHeight(), img.getWidth(),
			m_smoothingSigma, m_smoothingSigma);
		vl_dsift_process(filter, smoothedData);
		
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

ImageFeatures* FeatureExtractor::extractHog(Image& img) const {
	ImageFeatures* imageFeatures = new ImageFeatures(
			img.getWidth(), img.getHeight(), img.getNumChannels());
			
	unsigned int numStacks = 2;
	
	vector<pair<int, int> > coordinates;
	for(unsigned int x = m_gridSpacing / 2; x < img.getWidth();
			x += m_gridSpacing) {
			
		for(unsigned int y = m_gridSpacing / 2; y < img.getHeight();
				y += m_gridSpacing) {
				
			coordinates.push_back(make_pair(x, y));
		}
	}
	
	for(unsigned int i = 0; i < img.getNumChannels(); i++) {
		VlHog* hog = vl_hog_new(VlHogVariantUoctti, 9, false);
		vl_hog_put_image(hog, img.getData(i), img.getWidth(), img.getHeight(),
			1, m_gridSpacing);
		
		int descriptorSize = vl_hog_get_dimension(hog);
		int numDescriptors = vl_hog_get_width(hog) * vl_hog_get_height(hog);

		float* descriptors =
			new float[descriptorSize * numDescriptors];
		vl_hog_extract(hog, descriptors);
		
		unsigned int numDescX = (img.getWidth() + m_gridSpacing / 2)
			/ m_gridSpacing;
		unsigned int numDescY = (img.getHeight() + m_gridSpacing / 2)
			/ m_gridSpacing;
		float* newDescriptors = stackFeatures(descriptors, descriptorSize,
			numDescriptors,	numDescX, numDescY, numStacks);
	
		imageFeatures->addFeatures(i, newDescriptors,
			descriptorSize * numStacks,
			(numDescX - numStacks + 1) * (numDescY - numStacks + 1),
			coordinates);
	
		delete[] descriptors;
		delete[] newDescriptors;
		vl_hog_delete(hog);
	}
	
	return imageFeatures;
}
