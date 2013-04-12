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
		delete m_svmModel;
		m_svmModel = nullptr;
	}
	
	if(m_svmProb != nullptr) {
		for(int i = 0; i < m_svmProb->l; i++) {
			delete[] m_svmProb->x[i];
		}
		delete[] m_svmProb->x;
		
		//delete[] m_svmProb->y;
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
	m_svmParams->solver_type = linear::MCSVM_CS;
	m_svmParams->C = m_c;
	m_svmParams->eps = 1e-4;

	unsigned int descriptorLength = histograms[0]->getLength();
	linear::feature_node** kernel =
		new linear::feature_node*[histograms.size()];

	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < histograms.size(); i++) {
		kernel[i] = new linear::feature_node[descriptorLength + 1];
		for(unsigned int j = 0; j < descriptorLength; j++) {				
			kernel[i][j].index = j + 1;
			kernel[i][j].value = histograms[i]->getData()[j];
		}
		kernel[i][descriptorLength].index = -1;
		
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
	m_svmProb->n = descriptorLength;
	m_svmProb->y = &newClasses[0];
	m_svmProb->x = kernel;
	m_svmProb->bias = 0;

	m_svmModel = linear::train(m_svmProb, m_svmParams);
	cout << endl;
}

double LinearClassifier::test(vector<Histogram*> testHistograms,
		vector<unsigned int> testClasses) {
			
	OutputHelper::printMessage("Testing Classifier:");
	ConfusionMatrix confMat(m_classNames);
	
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < testHistograms.size(); i++) {
		pair<unsigned int, double> result = classify(testHistograms[i]);
		
		confMat.addEntry(testClasses[i], result.first);
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printResults("Predicting image", currentIter,
				testHistograms.size(), result.first, result.second);
		}
	}
	confMat.printMatrix();
	return confMat.getDiagonalAverage();
}

pair<unsigned int, double> LinearClassifier::classify(Histogram* histogram) {
	const unsigned int histLength = histogram->getLength();
	linear::feature_node testNode[histLength + 1];

	#pragma omp parallel for
	for(unsigned int j = 0; j < histLength; j++) {				
		testNode[j].index = j + 1;
		testNode[j].value = histogram->getData()[j];
	}
	testNode[histLength].index = -1;
	
	unsigned int predictedClass = linear::predict(m_svmModel, testNode);
	
	return make_pair(predictedClass, 0.0);
}
