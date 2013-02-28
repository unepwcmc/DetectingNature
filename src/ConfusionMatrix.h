#ifndef CONFUSION_MATRIX_H
#define CONFUSION_MATRIX_H

#include <boost/multi_array.hpp>

#include "OutputHelper.h"

class ConfusionMatrix {
public:
	ConfusionMatrix(std::vector<std::string> classNames);
	
	void addEntry(unsigned int originalClass, unsigned int predictedClass);
	void printMatrix();	
private:
	boost::multi_array<float, 2> m_confusionMatrix;
	std::vector<std::string> m_classNames;
	
	void normalize();
};

#endif
