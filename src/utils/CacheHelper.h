#ifndef CACHE_HELPER_H
#define CACHE_HELPER_H

#include <cctype>
#include <string>

#include <boost/range/algorithm/remove_if.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>

#include "framework/SettingsManager.h"
#include "features/ImageFeatures.h"
#include "utils/DatasetManager.h"


/**
 * @brief Helper for data serialization and deserialization.
 *
 * Provides utilities for loading and saving data, creating a unique cache
 * name based on the requested file and the classification parameters.
 */
class CacheHelper {
public:
	/**
	 * @brief Creates a @a CacheHelper class instance.
	 *
	 * @param datasetPath Relative path of the dataset we are parsing. This is
	 * used to keep data from different datasets in different cache folders.
	 * @param settings The application settings. Used to prevent false cache
	 * hits when different settings are used for the same dataset.
	 */
	CacheHelper(std::string datasetPath, const SettingsManager* settings);
	
	/**
	 * @brief Load the data from the hard drive.
	 *
	 * Generates a unique cache filepath and name based on the given @a filename
	 * and retrieves the data from cache, if it exists.
	 *
	 * @param filename Name that uniquely identifies the data to be loaded.
	 * Must be the same name used when saving the file.
	 * @return The requested data or @a nullptr if it is not cached
	 */
	template <typename T> T* load(std::string filename) const {
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
	
	/**
	 * @brief Save the data to the hard drive.
	 *
	 * Generates a unique cache filepath and name based on the given @a filename
	 * and saves the data to cache.
	 *
	 * @param filename Name that uniquely identifies the data to be saved.
	 * @param data The data item to be cached. This item must be serializable
	 * by the Boost Serialization library.
	 */
	template <typename T> void save(std::string filename, T* data) const {
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
	std::string m_datasetPath;
	const SettingsManager* m_settings;
	
	std::string getCacheFolder(std::string filename, 
		const std::type_info& dataType) const;
};

#endif
