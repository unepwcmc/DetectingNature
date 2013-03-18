#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

#include <boost/serialization/vector.hpp>

/**
 * @brief Stores the histogram of an image.
 *
 * This histogram is an intermediate image representation which can be used as
 * an input to a classifier.
 */
class Histogram {
public:
	/**
	 * @brief Initializes the histogram with its data.
	 *
	 * @param data The histogram data. This data is copied and can be
	 * safely deleted.
	 * @param length The size of the histogram.
	 */
	Histogram(double* data, unsigned int length);
	
	/**
	 * @brief Provides the stored size of the histogram.
	 *
	 * @return The size of the histogram.
	 */
	unsigned int getLength() const {
		return m_length;
	}
	
	/**
	 * @brief Provides the data of the histogram.
	 *
	 * @return The data of the histogram.
	 */
	const double* getData() const {
		return &m_data[0];
	}
	
private:
	std::vector<double> m_data;
	unsigned int m_length;
	
	// Boost serialization
	friend class boost::serialization::access;
	Histogram();
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_length;
		ar & m_data;
	}
};

#endif
