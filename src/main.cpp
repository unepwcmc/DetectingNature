#include <cstdio>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

typedef vector<path> pthvec;


// Returns the list of the relative paths of every file inside a folder.
pthvec list_files(string directory) {
	path dir(directory);

	pthvec filenames;
	for(directory_iterator it(dir); it != directory_iterator(); ++it) {
		filenames.push_back(it->path().relative_path());
	}
	
	return filenames;
}

int main(int argc, char** argv) {
	pthvec classes = list_files("data/scene_categories");
	for(pthvec::iterator it = classes.begin(); it != classes.end(); ++it) {
		cout << "Class: " << it->filename().string() << endl;
		pthvec filenames = list_files(it->string());
		for(pthvec::iterator it2 = filenames.begin(); it2 != filenames.end(); ++it2) {
			cout << it2->filename().string() << ",";
			cv::Mat image = cv::imread(it->string(), CV_LOAD_IMAGE_GRAYSCALE);
			
			//cv::namedWindow(*it, cv::CV_WINDOW_AUTOSIZE);
			//cv::imshow(*it, image);
		}
		cout << endl;
	}

	cv::waitKey(0);

	return 0;
}
