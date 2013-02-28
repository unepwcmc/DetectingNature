#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>

#include <boost/serialization/vector.hpp>

class Histogram {
public:
	Histogram();
	Histogram(float* data, unsigned int length);
	
	unsigned int getLength() const;
	const float* getData() const;
	
private:
	std::vector<float> m_data;
	unsigned int m_length;
	
	// Boost serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_length;
		ar & m_data;
	}
};

#endif
