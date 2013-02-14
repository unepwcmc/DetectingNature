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
		for(directory_iterator it(dir); it != directory_iterator(); it++) {
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


// Take the transformed dataset and train the classifier
CvSVM train_classifier(vector<string> class_list,
		map<string, cv::Mat> train_data) {
		
	// Train the classifier
	cout << "Training started" << endl;
	
	cv::Mat samples;
	cv::Mat labels(0, 1, CV_32FC1);
	cv::Mat class_labels;
	
	samples.push_back(train_data["manmade"]);
	class_labels = cv::Mat::zeros(train_data["manmade"].rows, 1, CV_32FC1);
	labels.push_back(class_labels);	
	samples.push_back(train_data["natural"]);
	class_labels = cv::Mat::ones(train_data["natural"].rows, 1, CV_32FC1);
	labels.push_back(class_labels);	
	
	cout << "Samples: " << samples << endl;
	cout << "Labels: " << labels << endl;
	
	//cv::Mat samples_32f; samples.convertTo(samples_32f, CV_32F);
	CvSVM classifier;
	classifier.train(samples, labels);
	
	//cv::FileStorage fs("classifier.xml", cv::FileStorage::WRITE);
	//fs << "classifier" << classifier;
	
	return classifier;
}


// Transform and prepare the dataset to be trained
CvSVM get_classifier(const string dataset,
		const unsigned int num_training_imgs, const cv::Mat& vocabulary) {
		
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	vector<string> class_list;
	map<string, cv::Mat> training_data;

	// Go trough all classes in the given dataset
	pthvec classes = list_files(dataset);
	BOOST_FOREACH(path cls, classes) {
		const string classname = cls.filename().string();
		
		cout << "Class: " << classname << endl;
		
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
		    	
		    	// Create dataset if if doesn't exist
		    	if(training_data.count(classname) == 0) {
		    		training_data[classname].create(
						0, img_descriptor.cols, img_descriptor.type());
					class_list.push_back(classname);
				}
		    	
		    	// Add this image to the class dataset
				training_data[classname].push_back(img_descriptor);
			}
		}
	}
	
	return train_classifier(class_list, training_data);
}


void test_classifier(const string dataset, const CvSVM& classifier,
		const cv::Mat& vocabulary) {
	
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	// Go trough all classes in the given dataset
	pthvec classes = list_files(dataset);
	BOOST_FOREACH(path cls, classes) {
		const string classname = cls.filename().string();
		
		cout << "Class: " << classname << endl;
		
		// Go trough each picture in the class
		pthvec filenames = list_files(cls.string());
		
		#pragma omp parallel for
		for(unsigned int i = 0;	i < filenames.size(); i++) {
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
		    	cout << img.filename().string() << ": ";
		    	
		    	cout << classifier.predict(img_descriptor) << endl;
			}
		}
	}
}


int main(int argc, char** argv) {
	// Define algorithm parameters
	const string dataset = "data/scene_categories";
	const unsigned int codebook_size = 400;
	const unsigned int num_training_images = 100;
	
	// Handle the vocabulary creation or loading
	cv::Mat vocabulary;
	if(exists("vocab.xml")) {
		cv::FileStorage fs("vocab.xml", cv::FileStorage::READ);
		fs["vocabulary"] >> vocabulary;
	} else {
		vocabulary = compute_codebook(dataset, codebook_size);
	}
	
	// Handle the classifier creation or loading
	CvSVM classifier = get_classifier(dataset, num_training_images, vocabulary);
	
	test_classifier(dataset, classifier, vocabulary);

	return 0;
}
