#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#include <libsvm/svm.h>

#include "classification/Classifier.h"
#include "framework/SettingsManager.h"
#include "classification/ConfusionMatrix.h"
#include "codebook/Histogram.h"

/**
 * @brief Trains a kernelized SVM for image classification.
 *
 * Trains several Support Vector Machine classifiers using a one-vs-all
 * technique to distinguish between several image classes.
 */
class SVMClassifier : public Classifier {
public:
	/**
	 * @brief Initializes the classifier with the required data.
	 *
	 * @param settings Manager that allows any required settings
	 * to be loaded from the configuration file.
	 * @param classNames A vector containing the names of all classes. Its
	 * order will remain unchanged and will be used to print the
	 * confusion matrix.
	 */
	SVMClassifier(const SettingsManager* settings,
		std::vector<std::string> classNames);
	~SVMClassifier();

	void train(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses);
		
	std::pair<unsigned int, double> classify(Histogram* histogram);

private:
	float m_c;
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
