#ifndef SETTINGS_MANAGER_H
#define SETTINGS_MANAGER_H

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

/**
 * @brief Stores all the classification parameters.
 *
 * These are used to configure all the algorithms used in the image
 * classification process. They are also used to generate the cache folder
 * names.
 *
 * @see CacheHelper
 */
class SettingsManager {
public:
	/**
	 * @brief Loads a new settings structure.
	 * 
	 * Uses the values defined in an XML file.
	 */
	SettingsManager(std::string filename) {
		boost::property_tree::read_json(filename, m_tree);
	}

	/**
	 * @brief Loads a value from the configuration data.
	 * 
	 * @param nodePath The identifier of the variable to be loaded
	 * @return The value defined in the configuration file
	 */
	template<typename T> T get(std::string nodePath) const {
		return m_tree.get<T>(nodePath);
	}

private:
	boost::property_tree::ptree m_tree;
};

#endif
