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
cv::Mat compute_codebook(const string dataset,
	const unsigned int codebook_size) {
	// Initialize Cluster Trainer
	cv::BOWKMeansTrainer codebook_trainer(codebook_size);


	// Go trough all classes in the given dataset
	pthvec filenames = list_files(dataset, true);
	
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < filenames.size(); i++) {
		path img = filenames[i];
		
		// Load the image
		cout << img.filename().string() << endl;
		cv::Mat input = cv::imread(img.string(), CV_LOAD_IMAGE_GRAYSCALE);
				
		// Extract features
		std::vector<cv::KeyPoint> keypoints;
		features_detector->detect(input, keypoints);
		
		cv::Mat features;
		descriptor_extractor->compute(input, keypoints, features);
		
		// Train bag-of-words
		#pragma omp critical 
		{
	    	codebook_trainer.add(features);
	    }
	}
	
	// Train the clusters
	cv::Mat vocabulary = codebook_trainer.cluster();
	//TODO Store vocabulary
	
	cv::FileStorage fs("vocab.xml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	
	return vocabulary;
}

void train_classifier(const string dataset,
	const unsigned int num_training_imgs, const cv::Mat& vocabulary) {
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	cv::Mat samples;
	cv::Mat labels;
	
	unsigned int class_idx = 0;

	// Go trough all classes in the given dataset
	pthvec classes = list_files(dataset);
	BOOST_FOREACH(path cls, classes) {
		cout << "Class: " << cls.filename().string() << endl;
		
		// Go trough each picture in the class
		pthvec filenames = list_files(cls.string());
		shuffle(filenames.begin(), filenames.end(),
			std::default_random_engine(0));
		
		#pragma omp parallel for
		for(unsigned int i = 0;
			i < min(num_training_imgs, filenames.size()); i++) {
			path img = filenames[i];
		
			// Load the image
			cv::Mat input = cv::imread(img.string(), CV_LOAD_IMAGE_GRAYSCALE);
			
			// Extract features
			std::vector<cv::KeyPoint> keypoints;
			features_detector->detect(input, keypoints);
			
			// Compute image final descriptor
			cv::Mat img_descriptor;
		    bow_extractor.compute(input, keypoints, img_descriptor);
		    
		    # pragma omp critical
		    {
		    	cout << img.filename().string() << endl;
				cout << img_descriptor << endl;
				samples.push_back(img_descriptor);
				labels.push_back(class_idx);
			}
		}
		class_idx++;
	}
	
	cout << "Labels: " << labels << endl;
	
	// Train the classifier
	cout << "Training started" << endl;
	CvSVM classifier;
	classifier.train(samples, labels);
}

int main(int argc, char** argv) {
	const unsigned int codebook_size = 100;
	const unsigned int num_training_images = 100;
	string dataset = "data/simple_scene_categories";
	
	cv::Mat vocabulary;
	
	if(exists("vocab.xml")) {
		cv::FileStorage fs("vocab.xml", cv::FileStorage::READ);
		fs["vocabulary"] >> vocabulary;
	} else {
		vocabulary = compute_codebook(dataset, codebook_size);
	}
	
	
	train_classifier(dataset, num_training_images, vocabulary);
	//test_classifier

	return 0;
}
