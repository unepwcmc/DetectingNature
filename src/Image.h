#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <opencv2/highgui/highgui.hpp>

class Image {
public:
	Image(std::string filename);
	~Image();
	
	unsigned int getWidth() const;
	unsigned int getHeight() const;
	float const* getData() const;

private:
	unsigned int m_width;
	unsigned int m_height;
	float* m_data;
};

#endif
