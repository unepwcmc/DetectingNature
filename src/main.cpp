#include <cstdio>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

#include <boost/assign/list_of.hpp>
using namespace boost::filesystem;

typedef vector<path> pthvec;

// Define descriptors to be used
const cv::Ptr<cv::FeatureDetector> features_detector =
	cv::FeatureDetector::create("SIFT");
const cv::Ptr<cv::DescriptorExtractor> descriptor_extractor =
	cv::DescriptorExtractor::create("SIFT");
const cv::Ptr<cv::DescriptorMatcher> descriptor_matcher =
	cv::DescriptorMatcher::create("FlannBased");


// Returns the list of the relative paths of every file inside a folder.
pthvec list_files(string directory, bool recursive=false) {
	
	path dir(directory);
	pthvec filenames;
		
	if(!recursive) {
		for(directory_iterator it(dir); it != directory_iterator(); ++it) {
			filenames.push_back(it->path().relative_path());
		}
	} else {
		for(recursive_directory_iterator it(dir);
			it != recursive_directory_iterator(); ++it) {

			if(is_regular_file(it->path()))
				filenames.push_back(it->path().relative_path());
		}
	}
	
	return filenames;
}


// Computes the codebook using the bag-of-words technique 
cv::Mat compute_codebook(const string dataset) {
	// Initialize Cluster Trainer
	cv::BOWKMeansTrainer codebook_trainer(100);


	// Go trough all classes in the given dataset
	pthvec filenames = list_files(dataset, true);
	BOOST_FOREACH(path img, filenames) {
		// Load the image
		cout << img.filename().string() << endl;
		cv::Mat input = cv::imread(img.string(), CV_LOAD_IMAGE_GRAYSCALE);
				
		// Extract features
		std::vector<cv::KeyPoint> keypoints;
		features_detector->detect(input, keypoints);
		
		cv::Mat features;
		descriptor_extractor->compute(input, keypoints, features);
		
		// Train bag-of-words
	    codebook_trainer.add(features);
	}
	
	// Train the clusters
	cv::Mat vocabulary = codebook_trainer.cluster();
	//TODO Store vocabulary
	
	return vocabulary;
}

void train_classifier(const string dataset,
	const int num_training_imgs, const cv::Mat& vocabulary) {
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	//TODO Init matrices
	cv::Mat samples;
	cv::Mat labels;
	
	int class_idx = 0;

	// Go trough all classes in the given dataset
	pthvec classes = list_files(dataset);
	BOOST_FOREACH(path cls, classes) {
		cout << "Class: " << cls.filename().string() << endl;
		
		int image_idx = 0;
		// Go trough each picture in the class
		pthvec filenames = list_files(cls.string());
		shuffle(filenames.begin(), filenames.end(),
			std::default_random_engine(0));
		BOOST_FOREACH(path img, filenames) {
			if(image_idx >= num_training_imgs) {
				break;
			}
		
			// Load the image
			cout << img.filename().string() << endl;
			cv::Mat input = cv::imread(img.string(), CV_LOAD_IMAGE_GRAYSCALE);
			
			// Extract features
			std::vector<cv::KeyPoint> keypoints;
			features_detector->detect(input, keypoints);
			
			// Compute image final descriptor
			cv::Mat img_descriptor;
		    bow_extractor.compute(input, keypoints, img_descriptor);
		    cout << img_descriptor << endl;
		    
		    samples.push_back(img_descriptor);
		    labels.push_back(class_idx);
		    
		    image_idx++;
		}
		class_idx++;
	}
	
	// Train the classifier
	cout << "Training started" << endl;
	CvSVM classifier;
	classifier.train(samples, labels);
}

int main(int argc, char** argv) {
	string dataset = "data/simple_scene_categories";
	cv::Mat vocabulary = compute_codebook(dataset);
	train_classifier(dataset, 100, vocabulary);
	//test_classifier

	return 0;
}
