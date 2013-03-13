#include "DatasetManager.h"
using namespace std;
using namespace boost;
using namespace boost::filesystem;

DatasetManager::DatasetManager() {
}

DatasetManager::DatasetManager(const string datasetPath,
		unsigned int trainImagesPerClass) {
		
	m_datasetPath = datasetPath;
	
	preloadFileLists();
	
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	shuffle(m_classFiles.begin(), m_classFiles.end(),
		default_random_engine(seed));
	sort(m_classNames.begin(), m_classNames.end());
	
	vector<unsigned int> classTotal(m_classNames.size(), 0);
	for(unsigned int i = 0; i < m_classFiles.size(); i++) {
		unsigned int classNumber = distance(m_classNames.begin(),
			find(m_classNames.begin(), m_classNames.end(),
			m_classFiles[i].second));
			
		if(classTotal[classNumber] < trainImagesPerClass) {
			m_trainFiles.push_back(m_classFiles[i].first);
			m_trainClasses.push_back(classNumber);
			classTotal[classNumber]++;
		} else {
			m_testFiles.push_back(m_classFiles[i].first);
			m_testClasses.push_back(classNumber);
		}
	}
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

std::vector<std::string> DatasetManager::getTrainData() const {
	return m_trainFiles;
}

std::vector<unsigned int> DatasetManager::getTrainClasses() const {
	return m_trainClasses;
}

std::vector<std::string> DatasetManager::getTestData() const {
	return m_testFiles;
}

std::vector<unsigned int> DatasetManager::getTestClasses() const {
	return m_testClasses;
}

string DatasetManager::getDatasetPath() const {
	return m_datasetPath;
}
