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
	Classifier(std::vector<std::string> classNames);
	~Classifier();
		
	void train(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses, double C);
	double test(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses);
	std::pair<unsigned int, double> classify(Histogram* histogram);

private:
	std::vector<Histogram*> m_trainHistograms;
	std::vector<double> m_trainClasses;
	std::vector<std::string> m_classNames;

	std::vector<svm_problem*> m_svmProbs;
	std::vector<svm_model*> m_svmModels;
	svm_parameter* m_svmParams;
		
	float* flattenHistogramData();
	double intersectionKernel(Histogram* a, Histogram* b);
	double* buildClassList(unsigned int desiredClass);
};

#endif
