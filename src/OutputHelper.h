#ifndef OUTPUT_HELPER_H
#define OUTPUT_HELPER_H

#include <cstdarg>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

#include <boost/multi_array.hpp>
#include <boost/algorithm/string.hpp>

/**
 * @brief Helper for printing progress messages and results.
 *
 * Allows for easy printing of messages over existing text, to prevent message
 * spam in the console. Also helps with the output of progress, including
 * formated progress percentage.
 *
 * @todo Allow for output messages to be toggled.
 */
class OutputHelper {
public:
	/**
	 * @brief Prints a single message, followed by a new line.
	 *
	 * @param message The message to be printed.
	 * @param indentLevel How many tab characters to insert at the beginning
	 * of the message.
	 */
	static void printMessage(std::string message = "",
		unsigned int indentLevel = 0);
	
	/**
	 * @brief Prints a single message, erasing the current line.
	 *
	 * Does not add a new line at the end of the message.
	 *
	 * @param message The message to be printed.
	 * @param indentLevel How many tab characters to insert at the beginning
	 * of the message.
	 */
	static void printInlineMessage(std::string message = "",
		unsigned int indentLevel = 0);
	
	/**
	 * @brief Shows the progress for a task.
	 *
	 * Automatically calculates the progress percentage and shows it along
	 * with the defined message.
	 * 
	 * @param message Message describing the current task.
	 * @param current The number of tasks that were completed.
	 * @param total The total number of tasks.
	 * @param indentLevel How many tab characters to insert at the beginning
	 * of the message.
	 */
	static void printProgress(std::string message, unsigned int current,
		unsigned int total,	unsigned int indentLevel = 1);
	
	/**
	 * @brief Shows the progress of the classification step.
	 *
	 * Automatically calculates the progress percentage and shows it along
	 * with the defined message, as well as the predicted class and decision
	 * value of the classification process.
	 * 
	 * @param message Message describing the current task.
	 * @param current The number of tasks that were completed.
	 * @param total The total number of tasks.
	 * @param result The chosen class for the classification of the image.
	 * @param value The decision value for the chosen class.
	 * @param indentLevel How many tab characters to insert at the beginning
	 * of the message.
	 */
	static void printResults(std::string message, unsigned int current,
		unsigned int total, int result, float value,
		unsigned int indentLevel = 1);
	
	/**
	 * @brief Prints a formatted confusion matrix.
	 *
	 * Also prints, below the matrix, the average value of the diagonal.
	 *
	 * @param classes The class labels to be printed along the rows and columns
	 * of the matrix.
	 * @param data the contents of the confusion matrix.
	 */
	static void printConfusionMatrix(const std::vector<std::string>& classes, 
		const boost::multi_array<float, 2>& data);

private:
	static std::string clearLine();
	
	static std::string indent(int level);
};

void printSvm(const char *s);
int printVlfeat(char const *format, ...);

#endif
