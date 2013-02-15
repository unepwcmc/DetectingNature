#include "DatasetManager.h"
using namespace std;
using namespace boost;
using namespace boost::filesystem;

DatasetManager::DatasetManager(const string datasetPath,
		const unsigned int numTrainImages) {
		
	m_datasetPath = datasetPath;
	m_numTrainImages = numTrainImages;
	
	preloadFileLists();
	
	random_shuffle(m_classFiles.begin(), m_classFiles.end());
}

DatasetManager::~DatasetManager() {
}

void DatasetManager::preloadFileLists() {
	path baseDir(m_datasetPath);
	for(directory_iterator it(baseDir); it != directory_iterator(); it++) {
		string className = it->path().filename().string();
		m_classNames.push_back(className);
		preloadClassFileList(className, it->path().relative_path());
	}
}

void DatasetManager::preloadClassFileList(string className, path classDir) {
	for(directory_iterator it(classDir); it != directory_iterator(); it++) {
		string fileName = it->path().relative_path().string();
		
		m_classFiles.push_back(make_pair(fileName, className));
	}
}

string DatasetManager::getFilename(string filePath) {
	return path(filePath).filename().string();
}

vector<string> DatasetManager::listClasses() {
	return m_classNames;
}

vector<string> DatasetManager::listFiles(DatasetPartitioning type) {
	vector<string> results;

	unsigned int totTrain = 0;
	pair<string, string> data;
	BOOST_FOREACH(data, m_classFiles) {
		if((type == ALL) ||
			(type == TRAIN && results.size() < m_numTrainImages) ||
			(type == TEST && totTrain >= m_numTrainImages)) {
			
			results.push_back(data.first);
		}
		
		if(type == TEST && totTrain < m_numTrainImages) {
			totTrain++;
		}
	}
	
	return results;
}

vector<string> DatasetManager::listFiles(
		DatasetPartitioning type, string desiredClass) {

	vector<string> results;

	unsigned int totTrain = 0;
	pair<string, string> data;
	BOOST_FOREACH(data, m_classFiles) {
		if(data.second == desiredClass) {
			if((type == ALL) ||
				(type == TRAIN && results.size() < m_numTrainImages) ||
				(type == TEST && totTrain >= m_numTrainImages)) {
				
				results.push_back(data.first);
			}
			
			if(type == TEST && totTrain < m_numTrainImages) {
				totTrain++;
			}
		}
	}
	return results;
}
