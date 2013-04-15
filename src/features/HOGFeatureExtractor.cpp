#include "HOGFeatureExtractor.h"
using namespace std;

HOGFeatureExtractor::HOGFeatureExtractor(const SettingsManager* settings) {
	m_gridSpacing = settings->get<vector<int> >("features.gridSpacing")[0];
	m_patchSize = settings->get<vector<int> >("features.patchSize")[0];
}

float* HOGFeatureExtractor::stackFeatures(float* descriptors,
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

ImageFeatures* HOGFeatureExtractor::extract(const ImageData* img) const {
	ImageFeatures* imageFeatures = new ImageFeatures(
		img->getWidth(), img->getHeight(), img->getNumChannels());
			
	unsigned int numStacks = 2;
	
	vector<pair<int, int> > coordinates;
	for(unsigned int x = m_gridSpacing / 2; x < img->getWidth();
			x += m_gridSpacing) {
			
		for(unsigned int y = m_gridSpacing / 2; y < img->getHeight();
				y += m_gridSpacing) {
				
			coordinates.push_back(make_pair(x, y));
		}
	}
	
	for(unsigned int i = 0; i < img->getNumChannels(); i++) {
		VlHog* hog = vl_hog_new(VlHogVariantUoctti, 9, false);
		vl_hog_put_image(hog, img->getData(i), img->getWidth(),
			img->getHeight(), 1, m_gridSpacing);
		
		int descriptorSize = vl_hog_get_dimension(hog);
		int numDescriptors = vl_hog_get_width(hog) * vl_hog_get_height(hog);

		float* descriptors =
			new float[descriptorSize * numDescriptors];
		vl_hog_extract(hog, descriptors);
		
		unsigned int numDescX = (img->getWidth() + m_gridSpacing / 2)
			/ m_gridSpacing;
		unsigned int numDescY = (img->getHeight() + m_gridSpacing / 2)
			/ m_gridSpacing;
		float* newDescriptors = stackFeatures(descriptors, descriptorSize,
			numDescriptors,	numDescX, numDescY, numStacks);
	
		if(i == 0) {
			imageFeatures->newFeatures(newDescriptors,
				descriptorSize * numStacks,
				(numDescX - numStacks + 1) * (numDescY - numStacks + 1),
				coordinates);
		} else {
			imageFeatures->extendFeatures(i, newDescriptors,
				(numDescX - numStacks + 1) * (numDescY - numStacks + 1));
		}
	
		delete[] descriptors;
		delete[] newDescriptors;
		vl_hog_delete(hog);
	}
	
	return imageFeatures;
}
