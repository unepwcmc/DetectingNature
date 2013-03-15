#include "CacheHelper.h"
using namespace std;

CacheHelper::CacheHelper(string datasetPath, Settings& settings) {
	m_datasetPath = datasetPath;
	m_settings = settings;
}

string CacheHelper::getCacheFolder(string filename,
		const type_info& dataType) {
	
	string basePath = "cache/" + m_datasetPath +
			"/" + dataType.name();
	
	if(dataType == typeid(DatasetManager)) {
		return basePath + "/";
	}
	
	stringstream cacheNameStream;
	cacheNameStream	<< 
		"_" << m_settings.colourspace <<
		"_" << m_settings.featureType <<
		"_" << m_settings.smoothingSigma << 
		"_" << m_settings.gridSpacing <<
		"_" << m_settings.patchSize;
		
	if(dataType == typeid(ImageFeatures)) {
		return basePath + cacheNameStream.str() + "/";
	}
	
	cacheNameStream	<<
		"_" << m_settings.textonImages <<
		"_" << m_settings.codewords <<
		"_" << m_settings.histogramType <<
		"_" << m_settings.pyramidLevels;
		
	
	return basePath + cacheNameStream.str() + "/";
}
