#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <map>

#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>

class DatasetManager {

public:
	enum DatasetPartitioning { ALL, TRAIN, TEST };

	DatasetManager(const std::string datasetPath,
		const unsigned int numTrainImages);
	~DatasetManager();
	
	static std::string getFilename(std::string filePath);
	
	std::vector<std::string> listClasses() const;
	std::vector<std::string> listFiles(DatasetPartitioning type) const;
	std::vector<std::string> listFiles(
		DatasetPartitioning type, std::string desiredClass) const;
		
private:
	void preloadFileLists();
	void preloadClassFileList(std::string className,
		boost::filesystem::path classDir);

	std::string m_datasetPath;
	unsigned int m_numTrainImages;
	std::vector<std::string> m_classNames;
	std::vector<std::pair<std::string, std::string> > m_classFiles;
};

#endif
