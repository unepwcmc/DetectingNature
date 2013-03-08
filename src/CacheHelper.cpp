#include "CacheHelper.h"
using namespace std;

CacheHelper::CacheHelper(Settings& settings) {
	m_settings = settings;
}

string CacheHelper::getCacheFolder(string filename,
		const type_info& dataType) {
	
	stringstream cacheNameStream;
	cacheNameStream	<< 
		"_" << m_settings.colourspace <<
		"_" << m_settings.featureType <<
		"_" << m_settings.gridSpacing <<
		"_" << m_settings.patchSize;
		
	if(dataType == typeid(ImageFeatures)) {
		return "cache/" + m_settings.datasetPath +
			"/" + dataType.name() + cacheNameStream.str() + "/";
	}
	
	cacheNameStream	<<
		"_" << m_settings.textonImages <<
		"_" << m_settings.codewords <<
		"_" << m_settings.pyramidLevels;
		
	if(dataType == typeid(Histogram)) {
		return "cache/" + m_settings.datasetPath +
			"/" + dataType.name() + cacheNameStream.str() + "/";
	}
	
	return "cache/" + m_settings.datasetPath +
			"/" + dataType.name() + "/";
}
