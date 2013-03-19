#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#include <libsvm/svm.h>

#include "classification/ConfusionMatrix.h"
#include "codebook/Histogram.h"

/**
 * @brief Contains the image classifier.
 *
 * Trains several Support Vector Machine using the one-vs-all technique to
 * distinguish between several image classes.
 */
class Classifier {
public:
	/**
	 * @brief Initializes the classifier with the required data.
	 *
	 * @param classNames A vector containing the names of all classes. Its
	 * order will remain unchanged and will be used to print the
	 * confusion matrix.
	 */
	Classifier(std::vector<std::string> classNames);
	~Classifier();
	
	/**
	 * @brief Trains the classifier.
	 *
	 * This will train as many classifiers as the number of classes, each one
	 * corresponding to one class versus every other image. The classifier with
	 * the highest response value is the one used as the classification for an
	 * image.
	 *
	 * @param histograms Histograms of the images to be used as training data.
	 * @param imageClasses Class of each image. Each element in this vector must
	 * match the element in the @a histograms vector at the same position.
	 * The class is a number that corresponds to its index in the @a classNames
	 * vector.
	 * @param C The C-SVM penalty value. Must be greater than 0. Values
	 * closer to 0 tend to underfit the data, while larger values tend
	 * to overfit.
	 */
	void train(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses, double C);
	
	/**
	 * @brief Tests the classifier using new images with a known class.
	 *
	 * This will print a confusion matrix. Each line of the matrix contains the
	 * percentage of images of that line's class that were classified as the
	 * column's class.
	 *
	 * @pre The classifiers must be trained using the train() function before
	 * this function can run.
	 * @param histograms Histograms of the images to be used as testing data.
	 * @param imageClasses Class of each image. Each element in this vector must
	 * match the element in the @a histograms vector at the same position.
	 * The class is a number that corresponds to its index in the @a classNames
	 * vector.
	 * @return The average value of the confusion matrix diagonal. Returned as a
	 * percentage, between 0 and 1.
	 */
	double test(std::vector<Histogram*> histograms,
		std::vector<unsigned int> imageClasses);
		
	/**
	 * @brief Classifies a single image.
	 *
	 * Calculates the most probable class for a given image using the previously
	 * trained classifiers.
	 *
	 * @param histogram Histogram of the image to be classified.
	 * @return A pair containing the chosen class index and the decision value
	 * of that class. The lower the decision value, the higher the chance of the
	 * image belonging to that class. Negative values mean at least one
	 * classifier thinks this image belongs to the chosen class, while a
	 * positive value means no classifier recognizes the image as belonging to
	 * their class.
	 */
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
