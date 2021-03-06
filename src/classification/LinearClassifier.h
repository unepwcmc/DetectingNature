#ifndef LINEAR_CLASSIFIER_H
#define LINEAR_CLASSIFIER_H

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

namespace linear {
	#include <linear.h>
}

#include "framework/SettingsManager.h"
#include "classification/Classifier.h"
#include "classification/ConfusionMatrix.h"
#include "codebook/Histogram.h"

/**
 * @brief Trains a linear SVM for image classification.
 *
 * Trains several Support Vector Machine classifiers using a one-vs-all
 * technique to distinguish between several image classes.
 */
class LinearClassifier : public Classifier {
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
	LinearClassifier(const SettingsManager* settings,
		std::vector<std::string> classNames);
	~LinearClassifier();
	
	void train(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses);

	std::pair<unsigned int, double> classify(Histogram* histogram);

private:
	float m_c;
	std::vector<std::string> m_classNames;

	linear::problem* m_svmProb;
	linear::model* m_svmModel;
	linear::parameter* m_svmParams;
	
	void clearData();
};

#endif
