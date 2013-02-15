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

#include "DatasetManager.h"

DatasetManager datasetManager("data/original_scene_categories", 100);


// Define descriptors to be used
const cv::Ptr<cv::FeatureDetector> features_detector =
	cv::FeatureDetector::create("SIFT");
const cv::Ptr<cv::DescriptorExtractor> descriptor_extractor =
	cv::DescriptorExtractor::create("SIFT");
const cv::Ptr<cv::DescriptorMatcher> descriptor_matcher =
	cv::DescriptorMatcher::create("FlannBased");


// Clear the current line to allow clean overwriting
string clear_line() {
	stringstream ss;
	ss << "\r";
	for(int i = 0; i < 80; i++)
		ss << " ";
	return ss.str();
}

string cleanFilename(string filename) {
	replace(filename.begin(), filename.end(), '/', '-');
	replace(filename.begin(), filename.end(), '.', '-');
	
	return filename;
}


// Computes the codebook using the bag-of-words technique 
cv::Mat compute_codebook(const unsigned int codebook_size) {
	
	cout << "Extracting Features:" << endl;
	
	// Initialize Cluster Trainer
	cv::BOWKMeansTrainer codebook_trainer(codebook_size);

	// Go trough all classes in the given dataset
	const vector<string> filenames =
		datasetManager.listFiles(DatasetManager::TRAIN);
	
	unsigned int totalProcessedImages = 0;
	
	#pragma omp parallel for
	for(unsigned int i = 0 ; i < filenames.size(); i++) {
		string img = filenames[i];
		
		// Load the image
		cv::Mat input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);
				
		// Extract features
		std::vector<cv::KeyPoint> keypoints;
		features_detector->detect(input, keypoints);
		
		cv::Mat features;
		descriptor_extractor->compute(input, keypoints, features);
		
		// Train bag-of-words
		#pragma omp critical
		{
			totalProcessedImages++;
		
	    	codebook_trainer.add(features);
	    
		    // Show progress
			int percent = totalProcessedImages * 100.0 / filenames.size();
			cout << "\r  Processing file " << DatasetManager::getFilename(img)
				<< " - " << totalProcessedImages << "/" << filenames.size()
				<< " (" << percent << "%)";
			cout.flush();
		}
	}
	cout << endl;
	
	// Train the clusters
	cout << "Computing Codebook:" << endl;
	cout << "  Clustering " << codebook_trainer.descripotorsCount()
		<< " descriptors";
	cout.flush();

	cv::Mat vocabulary = codebook_trainer.cluster();
	
	cv::FileStorage fs("vocab.xml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	
	cout << endl;
	
	return vocabulary;
}


// Take the transformed dataset and train the classifier
CvSVM train_classifier(vector<string> class_list,
		map<string, cv::Mat> train_data) {
		
	cout << "Training the classifier:" << endl;

	// Define samples and labels	
	cv::Mat samples;
	cv::Mat labels(0, 1, CV_32FC1);
	cv::Mat class_labels;
	
	vector<string> classes = datasetManager.listClasses();
	for(unsigned int i = 0; i < classes.size(); i++) {
		string classname = classes[i];
		samples.push_back(train_data[classname]);
		class_labels = cv::Mat(train_data[classname].rows, 1, CV_32FC1, i);
		labels.push_back(class_labels);
	}
	
	// Setup the classifier's parameters
	CvSVMParams classifier_params;
	classifier_params.svm_type = CvSVM::C_SVC;
	classifier_params.C = 1;
	classifier_params.kernel_type = CvSVM::RBF;
	classifier_params.gamma = 100;
	
	// Train the classifier
	CvSVM classifier;
	classifier.train(samples, labels, cv::Mat(), cv::Mat(), classifier_params);
	
	//TODO Save classifier
	//cv::FileStorage fs("classifier.xml", cv::FileStorage::WRITE);
	//fs << "classifier" << classifier;
	
	cout << clear_line() << "\r  Done" << endl;
	
	return classifier;
}


// Transform and prepare the dataset to be trained
CvSVM get_classifier(const cv::Mat& vocabulary) {
		
	cout << "Calculating feature histograms:" << endl;
	
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	vector<string> class_list;
	map<string, cv::Mat> training_data;
	
	cv::FileStorage fs("histograms.xml", cv::FileStorage::READ);

	// Go trough all classes in the given dataset
	const vector<string> classes = datasetManager.listClasses();
	BOOST_FOREACH(string classname, classes) {
		cout << "  Class: " << classname << endl;
				
		// Go through each picture in the class
		vector<string> filenames =
			datasetManager.listFiles(DatasetManager::TRAIN, classname);
		
		unsigned int totalProcessedImages = 0;
		#pragma omp parallel for
		for(unsigned int i = 0;	i < filenames.size(); i++) {
			string img = filenames[i];
		
			// Extract features
			cv::Mat img_descriptor;
			string imgName = cleanFilename(img);
			fs[imgName] >> img_descriptor;
		    
		    #pragma omp critical
		    {	    
			    totalProcessedImages++;
			    
		    	// Create dataset if it doesn't exist
		    	if(training_data.count(classname) == 0) {
		    		training_data[classname].create(
						0, img_descriptor.cols, img_descriptor.type());
					class_list.push_back(classname);
				}
		    	
		    	// Add this image to the class dataset
				training_data[classname].push_back(img_descriptor);

				// Show progress
				int percent = totalProcessedImages * 100.0 / filenames.size();
				cout << "\r    Processing file "
					<< DatasetManager::getFilename(img)
					<< " - " << totalProcessedImages << "/" << filenames.size()
					<< " (" << percent << "%)";
				cout.flush();
			}
		}
		cout << endl;
	}
	
	return train_classifier(class_list, training_data);
}


void test_classifier(const CvSVM& classifier, const cv::Mat& vocabulary) {
	
	cout << "Testing the classifier:" << endl;
	
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	cv::FileStorage fs("histograms.xml", cv::FileStorage::READ);
	
	// Go trough all classes in the given dataset
	vector<string> classes = datasetManager.listClasses();
	for(unsigned int i = 0; i < classes.size(); i++) {
		const string classname = classes[i];
		
		cout << "  Class: " << classname << endl;
		
		// Go trough each picture in the class
		vector<string> filenames =
			datasetManager.listFiles(DatasetManager::ALL, classname);
		vector<string> filenames_train =
			datasetManager.listFiles(DatasetManager::TRAIN, classname);
		vector<string> filenames_test =
			datasetManager.listFiles(DatasetManager::TEST, classname);
		
		unsigned int correct_classifications_train = 0;
		unsigned int correct_classifications_test = 0;
		
		unsigned int totalProcessedImages = 0;
		#pragma omp parallel for
		for(unsigned int j = 0;	j < filenames.size(); j++) {
			string img = filenames[j];
			
			cv::Mat img_descriptor;
			string imgName = cleanFilename(img);
			fs[imgName] >> img_descriptor;		    
		    
		    // Update results
		    const unsigned int class_result =
				classifier.predict(img_descriptor);
			const float class_result_value =
				classifier.predict(img_descriptor, true);
			
	    	if(class_result == i) {
	    		if(std::find(filenames_train.begin(),
	    			filenames_train.end(), img) != filenames_train.end()) {
		    		correct_classifications_train++;
		    	} else {
		    		correct_classifications_test++;
		    	}
		    }
		    
		    // Show progress
		    #pragma omp critical
		    {
		    	totalProcessedImages++;
		    	
		    	int percent = totalProcessedImages * 100.0 / filenames.size();
		    	
				cout << clear_line()
					<< "\r    Testing file " << DatasetManager::getFilename(img)
					<< " - " << totalProcessedImages << "/" << filenames.size()
					<< " (" << percent << "%)"
					<< " = Class " << class_result << " ("
		    		<< class_result_value << ")";
				cout.flush();
			}
		}
		
		// Determine the recognition and recall rates
		int percent_recall =
			correct_classifications_train * 100.0 / filenames_train.size();	
		
		int percent_recognition =
			correct_classifications_test * 100.0 / filenames_test.size();
		
		cout << clear_line()
			<< "\r    Recall: " << percent_recall << "%" << endl;
		cout << "    Recognition: " << percent_recognition << "%" << endl;
	}
}

void generateDescriptorCache(const cv::Mat& vocabulary) {
		
	cout << "Generating descriptor cache:" << endl;
	
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	// Go trough all classes in the given dataset
	vector<string> filenames =
		datasetManager.listFiles(DatasetManager::ALL);
	
	cv::FileStorage fs("histograms.xml", cv::FileStorage::WRITE);

	unsigned int totalProcessedImages = 0;
	#pragma omp parallel for
	for(unsigned int i = 0;	i < filenames.size(); i++) {
		string img = filenames[i];
	
		// Extract features
		cv::Mat input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<cv::KeyPoint> keypoints;
		features_detector->detect(input, keypoints);
		
		// Compute image final descriptor
		cv::Mat img_descriptor;
	    bow_extractor.compute(input, keypoints, img_descriptor);
	    
	    #pragma omp critical
	    {	    
		    totalProcessedImages++;
		    
		    string imgName = cleanFilename(img);		    
			fs << imgName << img_descriptor;

			// Show progress
			int percent = totalProcessedImages * 100.0 / filenames.size();
			cout << "\r    Processing file "
				<< DatasetManager::getFilename(img)
				<< " - " << totalProcessedImages << "/" << filenames.size()
				<< " (" << percent << "%)";
			cout.flush();
		}
	}
	cout << endl;
}


int main(int argc, char** argv) {
	// Define algorithm parameters
	const unsigned int codebook_size = 200;
	
	// Handle the vocabulary creation or loading
	cv::Mat vocabulary;
	if(exists("vocab.xml")) {
		cv::FileStorage fs("vocab.xml", cv::FileStorage::READ);
		fs["vocabulary"] >> vocabulary;
	} else {
		vocabulary = compute_codebook(codebook_size);
	}
	
	if(!exists("histograms.xml")) {
		generateDescriptorCache(vocabulary);
	}
	
	// Handle the classifier creation or loading
	CvSVM classifier = get_classifier(vocabulary);
	
	test_classifier(classifier, vocabulary);
	
	return 0;
}
