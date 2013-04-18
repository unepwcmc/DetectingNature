#include "SVMClassifier.h"
using namespace std;

SVMClassifier::SVMClassifier(const SettingsManager* settings,
		vector<string> classNames) {
		
	svm_set_print_string_function(&printSvm);
	
	m_c = settings->get<float>("classifier.c");
	m_classNames = classNames;
	m_svmParams = nullptr;
	m_svmProbs.resize(m_classNames.size(), nullptr);
	m_svmModels.resize(m_classNames.size(), nullptr);
}

SVMClassifier::~SVMClassifier() {
	if(m_svmProbs[0] != nullptr) {
		for(unsigned int i = 0; i < m_trainHistograms.size(); i++) {
			delete[] m_svmProbs[0]->x[i];
		}
		delete[] m_svmProbs[0]->x;
	}
	
	for(unsigned int i = 0; i < m_classNames.size(); i++) {
		if(m_svmModels[i] != nullptr) {
			delete m_svmModels[i];
		}
		
		if(m_svmProbs[i] != nullptr) {
			delete[] m_svmProbs[i]->y;
			delete m_svmProbs[i];
		}
	}

	if(m_svmParams != nullptr) {
		delete m_svmParams;
	}
}

float* SVMClassifier::flattenHistogramData() {
	int histLength = m_trainHistograms[0]->getLength();
	int datasetSize = m_trainHistograms.size() * histLength;
	float* data = new float[datasetSize];
	int currentIndex = 0;
	for(unsigned int i = 0; i < m_trainHistograms.size(); i++) {
		memcpy(&data[currentIndex], m_trainHistograms[i]->getData(), histLength);
		currentIndex += histLength;
	}
	return data;
}

inline double SVMClassifier::intersectionKernel(Histogram* a, Histogram* b) {
	double kernelVal = 0;	
	for(unsigned int i = 0; i < a->getLength(); i++) {
		kernelVal += min(a->getData()[i], b->getData()[i]);
	}
	return kernelVal;
}

double* SVMClassifier::buildClassList(unsigned int desiredClass) {
	double* classes = new double[m_trainHistograms.size()];
	for(unsigned int i = 0; i < m_trainHistograms.size(); i++) {
		classes[i] = (unsigned int)m_trainClasses[i] == desiredClass;
	}
	return classes;
}

void SVMClassifier::train(vector<Histogram*> histograms,
		vector<unsigned int> imageClasses) {
		
	m_trainHistograms = histograms;
	for_each(imageClasses.begin(), imageClasses.end(), [&](double imgClass) {
		m_trainClasses.push_back(imgClass);
	});
	
	OutputHelper::printMessage("Training Classifier:");

	m_svmParams = new svm_parameter();
	m_svmParams->svm_type = C_SVC;
	m_svmParams->kernel_type = PRECOMPUTED;
	m_svmParams->cache_size = 1000;
	m_svmParams->C = m_c;
	m_svmParams->eps = 1e-6;
	m_svmParams->shrinking = 1;
	m_svmParams->probability = 0;
	m_svmParams->nr_weight = 2;
	m_svmParams->weight_label = new int[2] {0, 1};
	m_svmParams->weight = new double[2]
		{100.0 / (histograms.size() - 100.0), 1.0};

	svm_node** kernel = new svm_node*[m_trainHistograms.size()];
	
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_trainHistograms.size(); i++) {
		kernel[i] = new svm_node[m_trainHistograms.size() + 2];
		kernel[i][0].index = 0;
		kernel[i][0].value = i + 1;
		for(unsigned int j = 0; j < m_trainHistograms.size(); j++) {				
			kernel[i][j+1].index = j + 1;
			kernel[i][j+1].value =
				intersectionKernel(m_trainHistograms[i], m_trainHistograms[j]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Calculating kernel matrix",
				currentIter, m_trainHistograms.size());
		}
	}
	
	for(unsigned int i = 0; i < m_classNames.size(); i++) {
		m_svmProbs[i] = new svm_problem();
		m_svmProbs[i]->l = m_trainHistograms.size();
		m_svmProbs[i]->y = buildClassList(i);
		m_svmProbs[i]->x = kernel;
	
		m_svmModels[i] = svm_train(m_svmProbs[i], m_svmParams);
		//svm_save_model("model.out", m_svmModel);
		cout << endl;
	}
}

pair<unsigned int, double> SVMClassifier::classify(Histogram* histogram) {
	svm_node testNode[m_trainHistograms.size() + 1];
	testNode[0].index = 0;
	testNode[0].value = 0;
	#pragma omp parallel for
	for(unsigned int j = 0; j < m_trainHistograms.size(); j++) {				
		testNode[j+1].index = j + 1;
		testNode[j+1].value =
			intersectionKernel(histogram, m_trainHistograms[j]);
	}
	
	unsigned int predictedClass = 0;
	double predictedValue = 1e6;
	
	for(unsigned int j = 0; j < m_classNames.size(); j++) {
		double thisValue;
		double thisClass =
			svm_predict_values(m_svmModels[j], testNode, &thisValue);
		thisValue = ((thisClass == 0 && thisValue < 0) ||
			(thisClass == 1 && thisValue > 0)) ?
			-thisValue : thisValue;
					
		if(thisValue < predictedValue) {
			predictedValue = thisValue;
			predictedClass = j;
		}			
	}
	
	return make_pair(predictedClass, predictedValue);
}
