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

	unsigned int classTotal[classNames.size()];
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
}

Classifier::~Classifier() {	
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
		double probabilities[m_imageClasses.size()];
		double predictedClass =
			svm_predict_probability(m_svmModel, testNode, probabilities);
		confMat.addEntry(m_imageClasses[i], predictedClass);
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printResults("Predicting image",
				currentIter, totalImages, predictedClass,
				*max_element(probabilities,
					probabilities + m_imageClasses.size()));
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

void Classifier::classify() {
	OutputHelper::printMessage("Training Classifier:");

	svm_problem* prob = new svm_problem();
	prob->l = m_numTrainImages;
	prob->y = &m_imageClasses[0];
	prob->x = new svm_node*[m_numTrainImages];
	
	unsigned int currentIter = 0;
	#pragma omp parallel for
	for(unsigned int i = 0; i < m_numTrainImages; i++) {
		prob->x[i] = new svm_node[m_numTrainImages + 2];
		prob->x[i][0].index = 0;
		prob->x[i][0].value = i + 1;
		#pragma omp parallel for
		for(unsigned int j = 0; j < m_numTrainImages; j++) {				
			prob->x[i][j+1].index = j + 1;
			prob->x[i][j+1].value =
				intersectionKernel(m_histograms[i], m_histograms[j]);
		}
		
		#pragma omp critical
		{
			currentIter++;
			OutputHelper::printProgress("Calculating kernel matrix",
				currentIter, m_numTrainImages);
		}
	}
	
	svm_parameter* params = new svm_parameter();
	params->svm_type = C_SVC;
	params->kernel_type = PRECOMPUTED;
	params->cache_size = 500;
	params->C = 10;
	params->eps = 1e-3;
	params->shrinking = 1;
	params->probability = 1;
	params->nr_weight = 0;
	
	m_svmModel = svm_train(prob, params);
	svm_save_model("model.out", m_svmModel);
	cout << endl;
}
