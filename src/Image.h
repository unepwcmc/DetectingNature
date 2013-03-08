#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <opencv2/highgui/highgui.hpp>

class Image {
public:
	enum Colourspace {GREYSCALE, OPPONENT};

	Image(std::string filename, Colourspace colour);
	~Image();
	
	unsigned int getNumChannels() const;
	unsigned int getWidth() const;
	unsigned int getHeight() const;
	float const* getData(unsigned int channel) const;

private:
	unsigned int m_width;
	unsigned int m_height;
	std::vector<float*> m_data;
};

#endif
