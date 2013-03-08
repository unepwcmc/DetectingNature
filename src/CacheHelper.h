#ifndef CACHE_HELPER_H
#define CACHE_HELPER_H

#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>

#include "Settings.h"
#include "ImageFeatures.h"
#include "DatasetManager.h"

class CacheHelper {
public:
	CacheHelper(Settings& settings);
	
	template <typename T> T* load(std::string filename) {
		std::string cacheFilename = getCacheFolder(filename, typeid(T)) +
			boost::replace_all_copy(filename, "/", "_");
		
		T* data = nullptr;
		if(boost::filesystem::exists(cacheFilename)) {
			std::ifstream ifs(cacheFilename);
			boost::archive::binary_iarchive ia(ifs);
			ia >> data;
		}
		return data;
	}
	
	template <typename T> void save(std::string filename, T* data) {
		std::string cacheFolder = getCacheFolder(filename, typeid(T));
		std::string cacheFilename = cacheFolder +
			boost::replace_all_copy(filename, "/", "_");
		
		if(!boost::filesystem::exists(cacheFolder)) {
			boost::filesystem::create_directories(cacheFolder);
		}
	
		std::ofstream ofs(cacheFilename);
		boost::archive::binary_oarchive oa(ofs);
		oa << data;
	}

private:
	Settings m_settings;
	
	std::string getCacheFolder(std::string filename, 
		const std::type_info& dataType);
};

#endif
