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

class OutputHelper {
public:	
	static void printMessage(std::string message = "",
		unsigned int indentLevel = 0);
	
	static void printInlineMessage(std::string message = "",
		unsigned int indentLevel = 0);
		
	static void printProgress(std::string message, unsigned int current,
		unsigned int total,	unsigned int indentLevel = 1);
		
	static void printResults(std::string message, unsigned int current,
		unsigned int total, int result, float value,
		unsigned int indentLevel = 1);
		
	static void printResults(unsigned int correctTrain, unsigned int sizeTrain,
		 unsigned int correctTest, unsigned int sizeTest);
	
	static void printConfusionMatrix(const std::vector<std::string>& classes, 
		const boost::multi_array<float, 2>& data);

	static std::string clearLine();
	
	static std::string indent(int level);
};

void printSvm(const char *s);
int printVlfeat(char const *format, ...);

#endif
