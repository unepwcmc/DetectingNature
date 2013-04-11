#include "HellingerFeatureTransform.h"
using namespace std;

HellingerFeatureTransform::HellingerFeatureTransform(
		const SettingsManager* settings) {
	
}

ImageFeatures* HellingerFeatureTransform::transform(
		const ImageFeatures* orig) const {
	
	ImageFeatures* transformedFeatures = new ImageFeatures(
		orig->getWidth(), orig->getHeight(), orig->getNumChannels());
	
	vector<pair<int, int> > coordinates;
	unsigned int descriptorSize = orig->getDescriptorSize();
	unsigned int numDescriptors = orig->getNumFeatures();
	float* newDescriptors = new float[numDescriptors * descriptorSize];
	for(unsigned int i = 0; i < numDescriptors; i++) {
		const float* descriptor = orig->getFeature(i);
		double l1norm = 1e-10;
		for(unsigned int j = 0; j < descriptorSize; j++) {
			l1norm += fabs(descriptor[j]);
		}
		
		for(unsigned int j = 0; j < descriptorSize; j++) {
			newDescriptors[i * descriptorSize + j] =
				sqrt(descriptor[j] / l1norm);
		}
		coordinates.push_back(orig->getCoordinates(i));
	}
	transformedFeatures->newFeatures(newDescriptors, descriptorSize,
		numDescriptors, coordinates);
	
	delete[] newDescriptors;
	delete orig;
	return transformedFeatures;
}
