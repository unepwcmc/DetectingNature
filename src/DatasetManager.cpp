#include "DatasetManager.h"
using namespace std;
using namespace boost;
using namespace boost::filesystem;

DatasetManager::DatasetManager() {
}

DatasetManager::DatasetManager(const string datasetPath) {
	m_datasetPath = datasetPath;
	
	preloadFileLists();
	
	random_shuffle(m_classFiles.begin(), m_classFiles.end());
	sort(m_classNames.begin(), m_classNames.end());
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

vector<string> DatasetManager::listClasses() const {
	return m_classNames;
}

vector<string> DatasetManager::listFiles() const {
	vector<string> results;

	pair<string, string> data;
	BOOST_FOREACH(data, m_classFiles) {
		results.push_back(data.first);
	}
	
	return results;
}

vector<unsigned int> DatasetManager::getImageClasses() const {
	vector<unsigned int> results;

	pair<string, string> data;
	BOOST_FOREACH(data, m_classFiles) {
		results.push_back(distance(m_classNames.begin(),
			find(m_classNames.begin(), m_classNames.end(), data.second)));
	}
	
	return results;
}

string DatasetManager::getDatasetPath() const {
	return m_datasetPath;
}
