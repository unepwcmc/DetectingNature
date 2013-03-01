#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#include <libsvm/svm.h>

#include "ConfusionMatrix.h"
#include "Histogram.h"

class Classifier {
public:
	Classifier(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses,
		std::vector<std::string> classNames, unsigned int trainImagesPerClass);
	~Classifier();
		
	void classify();
	void test();

private:
	std::vector<Histogram*> m_histograms;
	std::vector<double> m_imageClasses;
	std::vector<std::string> m_classNames;
	unsigned int m_numTrainImages;
		
	float* flattenHistogramData();
	double intersectionKernel(Histogram* a, Histogram* b);
	
	svm_model* m_svmModel;
	svm_parameter* m_svmParams;
	svm_problem* m_svmProb;
};

#endif
