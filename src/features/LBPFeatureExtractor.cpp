#include "LBPFeatureExtractor.h"
using namespace std;

LBPFeatureExtractor::LBPFeatureExtractor(const SettingsManager* settings) {
	m_gridSpacing = settings->get<vector<int> >("features.gridSpacing")[0];
	m_patchSize = settings->get<vector<int> >("features.patchSize")[0];
}

ImageFeatures* LBPFeatureExtractor::extract(const ImageData* img) const {
	ImageFeatures* imageFeatures = new ImageFeatures(
		img->getWidth(), img->getHeight(), img->getNumChannels());

	// Compute census transform
	vector<pair<int, int> > coordinates;
	for(unsigned int i = 0; i < img->getNumChannels(); i++) {
		unsigned int numTransforms =
			(img->getHeight() - 2) * (img->getWidth() - 2);
		float* transforms = new float[numTransforms * 8];
		
		unsigned int transIndex = 0;
		for(unsigned int y = 1; y < img->getHeight() - 1; y++) {
			for(unsigned int x = 1; x < img->getWidth() - 1; x++) {
				double centerValue = img->getData(i)[y * img->getWidth() + x];
			
				for(unsigned int dy = y - 1; dy <= y + 1; dy++) {
					for(unsigned int dx = x - 1; dx <= x + 1; dx++) {				
						if(dx == x && dy == y)
							continue;
						double neighbourValue =
							img->getData(i)[dy * img->getWidth() + dx];
						transforms[transIndex++] = neighbourValue > centerValue;
					}
				}
			}
		}
		
		// Group census transform into patches
		unsigned int numDescX =
			(img->getWidth() - m_patchSize - 2) / m_gridSpacing;
		unsigned int numDescY =
			(img->getHeight() - m_patchSize - 2) / m_gridSpacing;
		unsigned int descriptorSize = m_patchSize * m_patchSize;
		unsigned int numDescriptors = numDescX * numDescY;
		float* descriptors = new float[descriptorSize * numDescriptors * 8];
		unsigned int descIndex = 0;
		for(unsigned int y = 0; y < img->getHeight() - 2 - m_patchSize;
				y += m_gridSpacing) {
				
			for(unsigned int x = 0; x < img->getWidth() - 2 - m_patchSize;
					x += m_gridSpacing) {
					
				coordinates.push_back(
					make_pair(x + m_patchSize / 2, y + m_patchSize / 2));
					
				for(unsigned int dy = 0; dy < m_patchSize; dy++) {
					for(unsigned int dx = 0; dx < m_patchSize; dx++) {
						memcpy(&descriptors[descIndex], &transforms[
							(((x + dx) + (y + dy) * (img->getWidth() - 2)) * 8)],
							8 * sizeof(float));
						descIndex += 8;
					}
				}
			}
		}
		
		if(i == 0) {
			imageFeatures->newFeatures(descriptors, descriptorSize * 8,
				numDescriptors, coordinates);
		} else {
			imageFeatures->extendFeatures(i, descriptors, numDescriptors);
		}
		delete[] transforms;
		delete[] descriptors;
	}
			
	return imageFeatures;
}
