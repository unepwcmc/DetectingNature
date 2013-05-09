#include "LinearClassifier.h"
using namespace std;

LinearClassifier::LinearClassifier(const SettingsManager* settings,
		vector<string> classNames) {
		
	//linear::set_print_string_function(&printSvm);
	
	m_c = settings->get<float>("classifier.c");
	m_classNames = classNames;
	m_svmParams = nullptr;
	m_svmProb = nullptr;
	m_svmModel = nullptr;
}

LinearClassifier::~LinearClassifier() {
	clearData();
}

void LinearClassifier::clearData() {
	if(m_svmModel != nullptr) {
		linear::free_and_destroy_model(&m_svmModel);
		m_svmModel = nullptr;
	}
	
	if(m_svmProb != nullptr) {
		for(int i = 0; i < m_svmProb->l; i++) {
			delete[] m_svmProb->x[i];
		}
		delete[] m_svmProb->x;
		
		delete m_svmProb;
		m_svmProb = nullptr;
	}

	if(m_svmParams != nullptr) {
		delete m_svmParams;
		m_svmParams = nullptr;
	}
}

void LinearClassifier::train(vector<Histogram*> histograms,
		vector<unsigned int> imageClasses) {
	
	clearData();
	
	OutputHelper::printMessage("Training Classifier:");

	m_svmParams = new linear::parameter();
	m_svmParams->solver_type = linear::L2R_LR;
	m_svmParams->C = m_c;
	m_svmParams->eps = 1e-4;

	unsigned int descriptorLength = histograms[0]->getLength();
	linear::feature_node** kernel =
		new linear::feature_node*[histograms.size()];

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < histograms.size(); i++) {
		kernel[i] = new linear::feature_node[descriptorLength + 2];
		for(unsigned int j = 0; j < descriptorLength; j++) {				
			kernel[i][j].index = j + 1;
			kernel[i][j].value = histograms[i]->getData()[j];
		}
		kernel[i][descriptorLength].index = descriptorLength + 1;
		kernel[i][descriptorLength].value = 1.0;
		kernel[i][descriptorLength + 1].index = -1;
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Calculating kernel matrix",
				currentIter, histograms.size());
		}
	}
	
	vector<int> newClasses(imageClasses.begin(), imageClasses.end());
	m_svmProb = new linear::problem();
	m_svmProb->l = histograms.size();
	m_svmProb->n = descriptorLength + 1;
	m_svmProb->y = &newClasses[0];
	m_svmProb->x = kernel;
	m_svmProb->bias = 1;
	
	m_svmModel = linear::train(m_svmProb, m_svmParams);
	cout << endl;
}

pair<unsigned int, double> LinearClassifier::classify(Histogram* histogram) {
	const unsigned int histLength = histogram->getLength();
	linear::feature_node testNode[histLength + 2];

	#pragma omp parallel for
	for(unsigned int j = 0; j < histLength; j++) {				
		testNode[j].index = j + 1;
		testNode[j].value = histogram->getData()[j];
	}
	testNode[histLength].index = histLength + 1;
	testNode[histLength].value = 1.0;
	testNode[histLength + 1].index = -1;
	
	vector<double> probabilities(m_classNames.size(), 0.0);
	unsigned int predictedClass =
		linear::predict_probability(m_svmModel, testNode, &probabilities[0]);
	
	return make_pair(predictedClass,
		*max_element(probabilities.begin(), probabilities.end()));
}
