#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <map>
#include <random>
#include <chrono>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

/**
 * @brief Hold the information about the dataset.
 *
 * The class will index the files in a folder as well as associating them with
 * their class label. These files will also be split among two sets,
 * corresponding to the training data and the test data.
 */
class DatasetManager {
public:
	/**
	 * @brief Initializes the file and label lists.
	 *
	 * @pre This assumes the @a datasetPath points to a folder containing one
	 * subfolder per class, with the subfolder name being the class name. Each
	 * of these subfolders will contain all the images of that class.
	 * @warning There should be more than @a trainImagesPerClass images on
	 * each subfolder.
	 * 
	 * @param datasetPath The folder containing the dataset.
	 * @param trainImagesPerClass Number of images for each class that will
	 * be placed on the training set. The remainder will be used for testing.
	 */
	DatasetManager(std::string datasetPath,	unsigned int trainImagesPerClass);
	
	/**
	 * @brief Utility function that extracts the file name from a file path.
	 *
	 * @param filePath The path to be converted.
	 * @return The extracted file name.
	 */
	static std::string getFilename(std::string filePath) {
		return boost::filesystem::path(filePath).filename().string();
	}
	
	/**
	 * @brief Lists the name of all the classes in the dataset.
	 * 
	 * @return The vector with the class names, sorted alphabetically.
	 */
	std::vector<std::string> listClasses() const {
		return m_classNames;
	}
	
	/**
	 * @brief Lists the file paths of the train dataset.
	 *
	 * @return The shuffled list of files to be used on the classifier training.
	 */
	std::vector<std::string> getTrainData() const {
		return m_trainFiles;
	}
	
	/**
	 * @brief Lists the image classes of the train dataset.
	 *
	 * @return The list of the image classes for the training dataset
	 * in the same order as the getTrainData() function.
	 */
	std::vector<unsigned int> getTrainClasses() const {
		return m_trainClasses;
	}
	
	/**
	 * @brief Lists the file paths of the test dataset.
	 *
	 * @return The shuffled list of files to be used on the classifier testing.
	 */
	std::vector<std::string> getTestData() const {
		return m_testFiles;
	}
	
	/**
	 * @brief Lists the image classes of the test dataset.
	 *
	 * @return The list of the image classes for the testing dataset
	 * in the same order as the getTrainData() function.
	 */
	std::vector<unsigned int> getTestClasses() const {
		return m_testClasses;
	}
		
private:
	void preloadFileLists();
	void preloadClassFileList(std::string className,
		boost::filesystem::path classDir);
	std::vector<std::string> listFiles() const;
	std::vector<unsigned int> getImageClasses() const;

	std::string m_datasetPath;
	std::vector<std::string> m_classNames;
	std::vector<std::pair<std::string, std::string> > m_classFiles;
	
	std::vector<std::string> m_trainFiles;
	std::vector<unsigned int> m_trainClasses;
	std::vector<std::string> m_testFiles;
	std::vector<unsigned int> m_testClasses;
	
	// Boost serialization
	friend class boost::serialization::access;
	DatasetManager();
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_datasetPath;
		ar & m_classNames;
		ar & m_classFiles;
		ar & m_trainFiles;
		ar & m_trainClasses;
		ar & m_testFiles;
		ar & m_testClasses;
	}
};

#endif
