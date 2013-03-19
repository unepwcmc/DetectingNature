#ifndef CLASSIFICATION_FRAMEWORK_H
#define CLASSIFICATION_FRAMEWORK_H

#include <fstream>

#include "utils/CacheHelper.h"
#include "utils/DatasetManager.h"
#include "features/FeatureExtractor.h"
#include "codebook/CodebookGenerator.h"
#include "classification/Classifier.h"
#include "framework/Settings.h"

/**
 * @brief Main class for the classification of images.
 *
 * This class brings together all the scene classification bits ans pieces and
 * trains an image classifier capable of distinguishing between the different
 * categories in the given dataset.
 */
class ClassificationFramework {
public:
	/**
	 * @brief Initializes all the classification parameters
	 * 
	 * @param datasetPath Relative path to the dataset to be used to
	 * train the classifier.
	 * @param settings Contains the parameters used by the multiple algorithms
	 * involved in the classification process.
	 * @param skipCache If enabled the framework will ignore the cache when
	 * generating the codebook and the histograms. This does NOT regenerate
	 * the image descriptors since these are not random and should not change
	 * between multiple runs.
	 */
	ClassificationFramework(std::string datasetPath,
		Settings &settings, bool skipCache);
	~ClassificationFramework();
	
	/**
	 * @brief Trains and tests a classifier.
	 * 
	 * Splits the dataset into a training and a testing set, trains the
	 * classifier and returns the average accuracy of the classifier.
	 *
	 * @return The average of the diagonal of the confusion matrix. Returned as
	 * a percentage, between 0 and 1.
	 */
	double testRun();
	
	/**
	 * @brief Trains a classifier and predicts the class of other images.
	 * 
	 * Uses a subset of the dataset to train the classifier and the uses it to
	 * predict the classes for all images in @a imagesFolder.
	 *
	 * @param imagesFolder The folder containing unclassified images for wich
	 * we want to determine their class.
	 * @return A map relating the file path and the name of its predicted class.
	 */
	std::map<std::string, std::string> classify(std::string imagesFolder);

private:
	bool m_skipCache;
	Settings m_settings;
	CacheHelper* m_cacheHelper;
	DatasetManager* m_datasetManager;
	FeatureExtractor* m_featureExtractor;
	std::vector<std::string> m_imagePaths;
	std::string m_cachePath;
	
	Codebook* prepareCodebook(
		std::vector<std::string> imagePaths, bool skipCache);
	ImageFeatures* extractFeature(std::string imagePath);
	Histogram* generateHistogram(Codebook* codebook, std::string filePath);
	Classifier* trainClassifier(std::vector<Histogram*> trainHistograms); 
	
	std::vector<ImageFeatures*> extractFeatures(
		std::vector<std::string> imagePaths);
	std::vector<Histogram*> generateHistograms(
		std::vector<std::string> imagePaths, bool skipCodebook);
	double trainClassifier();
};

#endif
