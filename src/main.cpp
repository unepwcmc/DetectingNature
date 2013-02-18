#include <string>
using namespace std;

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
using namespace boost;

#include "TrainingSettings.h"
#include "CodebookManager.h"
#include "HistogramManager.h"

DatasetManager datasetManager("data/scene_categories", 100);
OutputHelper* m_outputHelper = new OutputHelper();


// Define descriptors to be used
const cv::Ptr<cv::DescriptorExtractor> descriptor_extractor =
	cv::DescriptorExtractor::create("SIFT");
const cv::Ptr<cv::DescriptorMatcher> descriptor_matcher =
	cv::DescriptorMatcher::create("FlannBased");


// Turn a filename into a valid XML tag name
string cleanFilename(string filename) {
	replace(filename.begin(), filename.end(), '/', '-');
	replace(filename.begin(), filename.end(), '.', '-');
	
	return filename;
}

// Take the transformed dataset and train the classifier
CvSVM train_classifier(map<string, cv::Mat> train_data) {
	m_outputHelper->printMessage("Training the classifier:");

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
	
	m_outputHelper->printMessage("Done:", 1);
	
	return classifier;
}


// Transform and prepare the dataset to be trained
CvSVM get_classifier(const cv::Mat& vocabulary) {
	
	m_outputHelper->printMessage("Calculating feature histograms:");
	
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
		m_outputHelper->printMessage("Class: " + classname, 1);
				
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
				m_outputHelper->printProgress("Processing file " +
					DatasetManager::getFilename(img),
					totalProcessedImages, filenames.size(), 2);
			}
		}
	}
	
	return train_classifier(training_data);
}


// Test the classififer against all images
void test_classifier(const CvSVM& classifier, const cv::Mat& vocabulary) {
	
	m_outputHelper->printMessage("Testing the classifier:");
	
	// Get new descriptors for images
	cv::BOWImgDescriptorExtractor bow_extractor(
		descriptor_extractor, descriptor_matcher);
		
	bow_extractor.setVocabulary(vocabulary);
	
	cv::FileStorage fs("histograms.xml", cv::FileStorage::READ);
	
	// Go trough all classes in the given dataset
	vector<string> classes = datasetManager.listClasses();
	multi_array<float, 2> confusionMatrix(
		extents[classes.size()][classes.size()]);
	fill(confusionMatrix.origin(),
		confusionMatrix.origin() + confusionMatrix.size(), 0);
	
	for(unsigned int i = 0; i < classes.size(); i++) {
		const string classname = classes[i];
		
		m_outputHelper->printMessage("Class: " + classname, 1);
		
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
		    
		    if(std::find(filenames_train.begin(),
	    			filenames_train.end(), img) == filenames_train.end()) {
				confusionMatrix[i][class_result]++;
			}
		    
		    // Show progress
		    #pragma omp critical
		    {
		    	totalProcessedImages++;
		    			    	
				m_outputHelper->printResults("Testing file " +
					DatasetManager::getFilename(img),
					totalProcessedImages, filenames.size(),
					class_result, class_result_value, 2);
			}
		}
		
		for(unsigned int j = 0; j < classes.size(); j++) {
			confusionMatrix[i][j] =
				confusionMatrix[i][j] / filenames_test.size();
		}
		
		// Show the recognition and recall rates
		m_outputHelper->printResults(
			correct_classifications_train, filenames_train.size(),
			correct_classifications_test, filenames_test.size());
	}
	
	m_outputHelper->printConfusionMatrix(classes, confusionMatrix);
}


int main() {
	// Define algorithm parameters
	TrainingSettings* settings = new TrainingSettings();
	settings->setDatasetSettings("data/scene_categories", 100);
	settings->setCodebookSettings(200, "SIFT", "SIFT");
	settings->setHistogramSettings("FlannBased");
	
	CodebookManager* codebookManager = new CodebookManager(settings);
	codebookManager->generateMissingVocabulary();
	cv::Mat vocabulary = codebookManager->getVocabulary();
	
	HistogramManager* histogramManager =
		new HistogramManager(settings, vocabulary);
	histogramManager->generateMissingHistograms();
	
	/*
	ClassifierManager* classifierManager =
		new ClassifierManager(settings, histogramManager);
	CvSVM classifier = classifierManager->classify();	
	*/
	
	// Handle the classifier creation or loading
	CvSVM classifier = get_classifier(vocabulary);
	
	test_classifier(classifier, vocabulary);
	
	return 0;
}
