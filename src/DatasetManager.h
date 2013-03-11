#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <map>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

class DatasetManager {

public:
	DatasetManager();
	DatasetManager(const std::string datasetPath,
		unsigned int trainImagesPerClass);
	
	static std::string getFilename(std::string filePath);
	
	std::vector<std::string> listClasses() const;
		
	std::vector<std::string> getTrainData() const;
	std::vector<unsigned int> getTrainClasses() const;
	
	std::vector<std::string> getTestData() const;
	std::vector<unsigned int> getTestClasses() const;
	
	std::string getDatasetPath() const;
		
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
