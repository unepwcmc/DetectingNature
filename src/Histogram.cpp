#include "Histogram.h"
using namespace std;

Histogram::Histogram() {
}

Histogram::Histogram(double* data, unsigned int length) {
	copy(data, data + length, back_inserter(m_data));
	m_length = length;
}

unsigned int Histogram::getLength() const {
	return m_length;
}

const double* Histogram::getData() const {
	return &m_data[0];
}
