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
	DatasetManager(const std::string datasetPath);
	~DatasetManager();
	
	static std::string getFilename(std::string filePath);
	
	std::vector<std::string> listClasses() const;
	std::vector<std::string> listFiles() const;
	std::vector<unsigned int> getImageClasses() const;
	std::string getDatasetPath() const;
	std::string getCachePath() const;
		
private:
	void preloadFileLists();
	void preloadClassFileList(std::string className,
		boost::filesystem::path classDir);

	std::string m_datasetPath;
	std::vector<std::string> m_classNames;
	std::vector<std::pair<std::string, std::string> > m_classFiles;
	
	// Boost serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version)
	{
		ar & m_datasetPath;
		ar & m_classNames;
		ar & m_classFiles;
	}
};

#endif
