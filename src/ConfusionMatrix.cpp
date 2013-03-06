#include "ConfusionMatrix.h"
using namespace std;

ConfusionMatrix::ConfusionMatrix(vector<string> classNames) {
	isNormalized = false;
	m_classNames = classNames;
	m_confusionMatrix.resize(
		boost::extents[classNames.size()][classNames.size()]);
}

void ConfusionMatrix::addEntry(
		unsigned int originalClass, unsigned int predictedClass) {

	m_confusionMatrix[originalClass][predictedClass]++;
	isNormalized = false;
}

void ConfusionMatrix::normalize() {
	for(unsigned int i = 0; i < m_classNames.size(); i++) {
		double lineTotal = 0;
		for(unsigned int j = 0; j < m_classNames.size(); j++) {
			lineTotal += m_confusionMatrix[i][j];
		}
		
		for(unsigned int j = 0; j < m_classNames.size(); j++) {
			m_confusionMatrix[i][j] /= lineTotal;
		}
	}
	isNormalized = true;
}

double ConfusionMatrix::getDiagonalAverage() {
	if(!isNormalized)
		normalize();

	double total = 0;
	for(unsigned int i = 0; i < m_classNames.size(); i++) {
		total += m_confusionMatrix[i][i];
	}
	return total / m_classNames.size();
}

void ConfusionMatrix::printMatrix() {
	if(!isNormalized)
		normalize();
	OutputHelper::printConfusionMatrix(m_classNames, m_confusionMatrix);
}
