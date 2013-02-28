#include "ConfusionMatrix.h"
using namespace std;

ConfusionMatrix::ConfusionMatrix(vector<string> classNames) {
	m_classNames = classNames;
	m_confusionMatrix.resize(
		boost::extents[classNames.size()][classNames.size()]);
}

void ConfusionMatrix::addEntry(
		unsigned int originalClass, unsigned int predictedClass) {

	m_confusionMatrix[originalClass][predictedClass]++;
}

void ConfusionMatrix::normalize() {
	for(unsigned int i = 0; i < m_classNames.size(); i++) {
		unsigned int lineTotal = 0;
		for(unsigned int j = 0; j < m_classNames.size(); j++) {
			lineTotal += m_confusionMatrix[i][j];
		}
		
		for(unsigned int j = 0; j < m_classNames.size(); j++) {
			m_confusionMatrix[i][j] /= lineTotal;
		}
	}
}

void ConfusionMatrix::printMatrix() {
	normalize();
	OutputHelper::printConfusionMatrix(m_classNames, m_confusionMatrix);
}
