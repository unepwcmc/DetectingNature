#include "CodebookGenerator.h"
using namespace std;

CodebookGenerator::CodebookGenerator(const SettingsManager* settings) {
	m_numFeatures = settings->get<unsigned int>("codebook.totalFeatures");
}

vector<float> CodebookGenerator::generateDescriptorSet(
		vector<ImageFeatures*> imageFeatures) const {
		
	int descriptorSize = imageFeatures[0]->getDescriptorSize();
	vector<float> descriptors;
	
	unsigned int numDescriptors = 0;
	for(unsigned int i = 0; i < imageFeatures.size(); i++) {
		numDescriptors += imageFeatures[i]->getNumFeatures();
	}
	
	double odds = m_numFeatures / (double)numDescriptors;
	for(unsigned int i = 0; i < imageFeatures.size(); i++) {
		for(unsigned int j = 0; j < imageFeatures[i]->getNumFeatures(); j++) {
			if(((double)rand()/(double)RAND_MAX) < odds) {
				const float* feature = imageFeatures[i]->getFeature(j);
				copy(feature, feature + descriptorSize,
					back_inserter(descriptors));
			}
		}
	}
	
	return descriptors;
}
