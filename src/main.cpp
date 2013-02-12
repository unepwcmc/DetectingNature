#include <cstdio>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <opencv2/opencv.hpp>
using namespace cv;

#include <boost/lambda/lambda.hpp>
using namespace boost::lambda;

int main(int argc, char** argv)
{
	// Test OpenCV
	Mat image = imread("data/demo.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	namedWindow("Display Image", CV_WINDOW_AUTOSIZE);
	imshow("Display Image", image);

	waitKey(0);

	// Test Boost
	typedef std::istream_iterator<int> in;

	std::for_each(
		in(std::cin), in(), std::cout << (_1 * 3) << " " );


	return 0;
}
