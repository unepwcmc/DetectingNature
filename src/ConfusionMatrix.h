#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

#include <boost/multi_array.hpp>

#include "OutputHelper.h"

/**
 * @brief Gathers and shows the results of the image classification process.
 *
 * After adding all the results of the testing of a classifier, this matrix,
 * relating the image's predicted class and the expected result, will be
 * normalized.
 * Each element of the matrix represents the percentage of images of a class
 * (the row) got classified as another class (the column). The diagonal shows
 * the percentage of correctly classified images.
 */
class ConfusionMatrix {
public:
	/**
	 * @brief Initializes the confusion matrix.
	 *
	 * @param classNames A vector with the names of each class. This vector
	 * will not be sorted and the resulting matrix will appear the same order as the one defined here.
	 */
	ConfusionMatrix(std::vector<std::string> classNames);
	
	/**
	 * @brief Adds one result to the matrix.
	 *
	 * The classes should be the index to the @a classNames vector defined
	 * when the confusion matrix was constructed.
	 *
	 * @warning Once the diagonal average or the final matrix is printed, the
	 * data will be normalized and no more data should be added using
	 * this function.
	 *
	 * @param originalClass The expected result.
	 * @param predictedClass The result obtained by using the classifier.
	 */
	void addEntry(unsigned int originalClass, unsigned int predictedClass);
	
	/**
	 * @brief Calculates the average of the matrix diagonal.
	 *
	 * This value is used instead of the overall correct classification ratio
	 * in order to prevent large or unbalanced classes from dominating the
	 * results.
	 *
	 * @return The average value of the matrix diagonal. This is a percentage,
	 * between 0 and 1.
	 */
	double getDiagonalAverage();
	
	/**
	 * @brief Prints the confusion matrix to the standard output.
	 */
	void printMatrix();	
private:
	boost::multi_array<float, 2> m_confusionMatrix;
	std::vector<std::string> m_classNames;
	bool isNormalized;
	
	void normalize();
};

#endif
