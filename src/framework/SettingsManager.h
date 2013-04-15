#ifndef SETTINGS_MANAGER_H
#define SETTINGS_MANAGER_H

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
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
	template<typename T>
	T get(std::string nodePath) const {
		return getImpl(nodePath, static_cast<T*>(0));
	}

private:
	boost::property_tree::ptree m_tree;
	
	template<typename T>
	T getImpl(const std::string& nodePath, T*) const {
		return m_tree.get<T>(nodePath);
	}
	
	template<typename T>
	std::vector<T> getImpl(
			const std::string& nodePath, std::vector<T>*) const {
		
		std::vector<T> result;
		BOOST_FOREACH(const boost::property_tree::ptree::value_type &v,
				m_tree.get_child(nodePath)) {
		
			result.push_back(boost::lexical_cast<T>(v.second.data()));
		}
		return result;
	}
};

#endif
