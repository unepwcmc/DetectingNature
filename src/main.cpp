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


// Clear the current line to allow clean overwriting
string clear_line() {
	stringstream ss;
	ss << "\r";
	for(int i = 0; i < 80; i++)
		ss << " ";
	return ss.str();
}


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
	
	cout << "Extracting Features:" << endl;
	
	// Initialize Cluster Trainer
	cv::BOWKMeansTrainer codebook_trainer(codebook_size);

	// Go trough all classes in the given dataset
	pthvec filenames = list_files(dataset, true);
	
	#pragma omp parallel for ordered
	for(unsigned int i = 0 ; i < filenames.size(); i++) {
		path img = filenames[i];
		
		// Load the image
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
	    
	    // Show progress
	    #pragma omp ordered
		{
			int percent = (i+1) * 100.0 / filenames.size();
			cout << "\r  Processing file " << img.filename().string()
				<< " - " << (i+1) << "/" << filenames.size()
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
	
	samples.push_back(train_data["natural"]);
	class_labels = cv::Mat::zeros(train_data["natural"].rows, 1, CV_32FC1);
	labels.push_back(class_labels);	
	samples.push_back(train_data["manmade"]);
	class_labels = cv::Mat::ones(train_data["manmade"].rows, 1, CV_32FC1);
	labels.push_back(class_labels);	
	
	// Setup the classifier's parameters
	CvSVMParams classifier_params;
	classifier_params.svm_type = CvSVM::NU_SVC;
	classifier_params.nu = 0.2;
	classifier_params.kernel_type = CvSVM::RBF;
	classifier_params.gamma = 10;
	
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
CvSVM get_classifier(const string dataset,
		const unsigned int num_training_imgs, const cv::Mat& vocabulary) {
		
	cout << "Calculating feature histograms:" << endl;
	
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
		cout << "  Class: " << classname << endl;
				
		// Go trough each picture in the class
		pthvec filenames = list_files(cls.string());
		//TODO Find a good way to shuffle the files
		//shuffle(filenames.begin(), filenames.end(),
		//	std::default_random_engine(0));
			
		const unsigned int class_num_images =
			min(num_training_imgs, filenames.size());
		
		#pragma omp parallel for ordered
		for(unsigned int i = 0;	i < class_num_images; i++) {
			path img = filenames[i];
		
			// Extract features
			cv::Mat input = cv::imread(img.string(), CV_LOAD_IMAGE_GRAYSCALE);		
			std::vector<cv::KeyPoint> keypoints;
			features_detector->detect(input, keypoints);
			
			// Compute image final descriptor
			cv::Mat img_descriptor;
		    bow_extractor.compute(input, keypoints, img_descriptor);
		    
		    #pragma omp critical
		    {	    	
		    	// Create dataset if if doesn't exist
		    	if(training_data.count(classname) == 0) {
		    		training_data[classname].create(
						0, img_descriptor.cols, img_descriptor.type());
					class_list.push_back(classname);
				}
		    	
		    	// Add this image to the class dataset
				training_data[classname].push_back(img_descriptor);
			}
			
			// Show progress
			#pragma omp ordered
			{
				int percent = (i+1) * 100.0 / class_num_images;
				cout << "\r    Processing file " << img.filename().string()
					<< " - " << (i+1) << "/" << class_num_images
					<< " (" << percent << "%)";
				cout.flush();
			}
		}
		cout << endl;
	}
	
	return train_classifier(class_list, training_data);
}


void test_classifier(const string dataset, const CvSVM& classifier,
		const cv::Mat& vocabulary, const unsigned int num_training_images) {
	
	cout << "Testing the classifier:" << endl;
	
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	// Go trough all classes in the given dataset
	pthvec classes = list_files(dataset);
	for(unsigned int i = 0; i < classes.size(); i++) {
		path cls = classes[i];
		const string classname = cls.filename().string();
		
		cout << "  Class: " << classname << endl;
		
		// Go trough each picture in the class
		pthvec filenames = list_files(cls.string());
		
		int correct_classifications_train = 0;
		int correct_classifications_test = 0;
		
		#pragma omp parallel for ordered
		for(unsigned int j = 0;	j < filenames.size(); j++) {
			path img = filenames[j];
		
			// Extract features
			cv::Mat input = cv::imread(img.string(), CV_LOAD_IMAGE_GRAYSCALE);
			std::vector<cv::KeyPoint> keypoints;
			features_detector->detect(input, keypoints);
			
			// Compute image final descriptor
			cv::Mat img_descriptor;
		    bow_extractor.compute(input, keypoints, img_descriptor);
		    
		    
		    // Update results
		    const unsigned int class_result =
				classifier.predict(img_descriptor);
			const float class_result_value =
				classifier.predict(img_descriptor, true);	
			
	    	if(class_result == i) {
	    		if(j < num_training_images)
		    		correct_classifications_train++;
		    	else
		    		correct_classifications_test++;
		    }
		    
		    // Show progress
		    #pragma omp ordered
		    {
		    	int percent = (j+1) * 100.0 / filenames.size();
		    	
				cout << clear_line()
					<< "\r    Testing file " << img.filename().string()
					<< " - " << (j+1) << "/" << filenames.size()
					<< " (" << percent << "%)"
					<< " = Class " << class_result << " ("
		    		<< class_result_value << ")";
				cout.flush();
			}
		}
		
		// Determine the recognition and recall rates
		int percent_recall =
			correct_classifications_train * 100.0 / num_training_images;	
		
		int percent_recognition =
			correct_classifications_test * 100.0 /
			(filenames.size() - num_training_images);
		
		cout << clear_line()
			<< "\r    Recall: " << percent_recall << "%" << endl;
		cout << "    Recognition: " << percent_recognition << "%" << endl;
	}
}


int main(int argc, char** argv) {
	// Define algorithm parameters
	const string dataset = "data/medium_scene_categories";
	const unsigned int codebook_size = 200;
	const unsigned int num_training_images = 20;
	
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
	
	test_classifier(dataset, classifier, vocabulary, num_training_images);

	return 0;
}
