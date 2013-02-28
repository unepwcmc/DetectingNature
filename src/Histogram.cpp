#include "Histogram.h"
using namespace std;

Histogram::Histogram() {
}

Histogram::Histogram(float* data, unsigned int length) {
	copy(data, data + length, back_inserter(m_data));
	m_length = length;
}

unsigned int Histogram::getLength() const {
	return m_length;
}

const float* Histogram::getData() const {
	return &m_data[0];
}
