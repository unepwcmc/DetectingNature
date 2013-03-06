#include "Classifier.h"
using namespace std;

Classifier::Classifier(vector<Histogram*> histograms,
		vector<unsigned int> imageClasses, vector<string> classNames,
		unsigned int trainImagesPerClass) {

	svm_set_print_string_function(&printSvm);

	std::vector<Histogram*> histogramsTrain;
	std::vector<unsigned int> imageClassesTrain;
	std::vector<Histogram*> histogramsTest;
	std::vector<unsigned int> imageClassesTest;

	vector<unsigned int> classTotal(classNames.size(), 0);
	for(unsigned int i = 0; i < histograms.size(); i++) {
		if(classTotal[imageClasses[i]] < trainImagesPerClass) {
			histogramsTrain.push_back(histograms[i]);
			imageClassesTrain.push_back(imageClasses[i]);
			classTotal[imageClasses[i]]++;
		} else {
			histogramsTest.push_back(histograms[i]);
			imageClassesTest.push_back(imageClasses[i]);
		}
	}
	
	copy(histogramsTrain.begin(), histogramsTrain.end(),
		back_inserter(m_histograms));
	copy(histogramsTest.begin(), histogramsTest.end(),
		back_inserter(m_histograms));
	copy(imageClassesTrain.begin(), imageClassesTrain.end(),
		back_inserter(m_imageClasses));
	copy(imageClassesTest.begin(), imageClassesTest.end(),
		back_inserter(m_imageClasses));
	
	m_classNames = classNames;
	m_numTrainImages = trainImagesPerClass * classNames.size();
	
	m_svmProb = nullptr;
	m_svmParams = nullptr;
	m_svmModels.resize(m_classNames.size(), nullptr);
}

Classifier::~Classifier() {
	if(m_svmProb != nullptr) {
		for(unsigned int i = 0; i < m_numTrainImages; i++)
			delete[] m_svmProb->x[i];
		delete[] m_svmProb->x;
		delete m_svmProb;
	}
	
	if(m_svmParams != nullptr) {
		delete m_svmParams;
	}
}

float* Classifier::flattenHistogramData() {
	int histLength = m_histograms[0]->getLength();
	int datasetSize = m_histograms.size() * histLength;
	float* data = new float[datasetSize];
	int currentIndex = 0;
	for(unsigned int i = 0; i < m_histograms.size(); i++) {
		memcpy(&data[currentIndex], m_histograms[i]->getData(), histLength);
		currentIndex += histLength;
	}
	return data;
}

void Classifier::test() {
	OutputHelper::printMessage("Testing Classifier:");
	ConfusionMatrix confMat(m_classNames);
	
	unsigned int totalImages = m_histograms.size() - m_numTrainImages;
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = m_numTrainImages; i < m_histograms.size(); i++) {
		svm_node testNode[m_numTrainImages];
		testNode[0].index = 0;
		testNode[0].value = 0;
		#pragma omp parallel for
		for(unsigned int j = 0; j < m_numTrainImages; j++) {				
			testNode[j+1].index = j + 1;
			testNode[j+1].value =
				intersectionKernel(m_histograms[i], m_histograms[j]);
		}
		
		double predictedClass = 0;
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
		
		confMat.addEntry(m_imageClasses[i], predictedClass);
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printResults("Predicting image",
				currentIter, totalImages, predictedClass, predictedValue);
		}
	}
	confMat.printMatrix();
}

double Classifier::intersectionKernel(Histogram* a, Histogram* b) {
	double kernelVal = 0;
	for(unsigned int i = 0; i < a->getLength(); i++) {
		kernelVal += min(a->getData()[i], b->getData()[i]);
	}
	return kernelVal;
}

double* Classifier::buildClassList(unsigned int desiredClass) {
	double* classes = new double[m_numTrainImages];
	for(unsigned int i = 0; i < m_numTrainImages; i++) {
		classes[i] = (unsigned int)m_imageClasses[i] == desiredClass;
	}
	return classes;
}

void Classifier::classify(double C) {
	OutputHelper::printMessage("Training Classifier:");

	m_svmParams = new svm_parameter();
	m_svmParams->svm_type = C_SVC;
	m_svmParams->kernel_type = PRECOMPUTED;
	m_svmParams->cache_size = 1000;
	m_svmParams->C = C;
	m_svmParams->eps = 1e-6;
	m_svmParams->shrinking = 1;
	m_svmParams->probability = 1;
	m_svmParams->nr_weight = 0;

	svm_node** kernel = new svm_node*[m_numTrainImages];
	
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_numTrainImages; i++) {
		kernel[i] = new svm_node[m_numTrainImages + 2];
		kernel[i][0].index = 0;
		kernel[i][0].value = i + 1;
		#pragma omp parallel for
		for(unsigned int j = 0; j < m_numTrainImages; j++) {				
			kernel[i][j+1].index = j + 1;
			kernel[i][j+1].value =
				intersectionKernel(m_histograms[i], m_histograms[j]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Calculating kernel matrix",
				currentIter, m_numTrainImages);
		}
	}
	
	for(unsigned int i = 0; i < m_classNames.size(); i++) {
		m_svmProb = new svm_problem();
		m_svmProb->l = m_numTrainImages;
		m_svmProb->y = buildClassList(i);
		m_svmProb->x = kernel;
	
		m_svmModels[i] = svm_train(m_svmProb, m_svmParams);
		//svm_save_model("model.out", m_svmModel);
		cout << endl;
	}
}
