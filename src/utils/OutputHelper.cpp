#include "OutputHelper.h"
using namespace std;
using namespace boost;

void printSvm(const char *s) {
	std::string str(s);
	if(str.find("optimization") != std::string::npos) {
		boost::erase_all(str, "\n");
		OutputHelper::printInlineMessage("Generating model - " + str, 1);
	}
}

int printVlfeat(char const *format, ...) {
	char dest[255];
	va_list argptr;
	va_start(argptr, format);
	vsnprintf(dest, 255, format, argptr);
	va_end(argptr);
	
	string str(&dest[19]);
	if(str.find("energy") != string::npos) {
		boost::erase_all(str, "\n");
		OutputHelper::printInlineMessage(
			"Clustering features (iteration " + str + ")", 1);
	} else if(str.find("because") != string::npos) {
		cout << endl;
	}
	return 0;
}

void OutputHelper::printInlineMessage(string message,
		unsigned int indentLevel) {
		
	cout << clearLine() << indent(indentLevel) << message;
	cout.flush();
}

void OutputHelper::printMessage(string message,
		unsigned int indentLevel) {
	
	printInlineMessage(message, indentLevel);
	cout << endl;
}

void OutputHelper::printProgress(const string message, unsigned int current,
		unsigned int total,	unsigned int indentLevel) {

	int percent = current * 100.0 / total;
	cout << clearLine() << indent(indentLevel) << message
		<< " - " << current << "/" << total
		<< " (" << percent << "%)";
	
	if(current == total)
		cout << endl;
	else
		cout.flush();
}

void OutputHelper::printResults(string message, unsigned int current,
		unsigned int total, int result, float value,
		unsigned int indentLevel) {
		
	int percent = current * 100.0 / total;
	cout << clearLine() << indent(indentLevel) << message
		<< " - " << current << "/" << total
		<< " (" << percent << "%)"
		<< " = Class " << result << " (" << value << ")";

	if(current == total)
		cout << endl;
	else
		cout.flush();
}

void OutputHelper::printConfusionMatrix(const vector<string>& classes,
		const multi_array<float, 2>& confusionMatrix) {
	
	int largestClassName = max_element(classes.begin(), classes.end(),
		[](string a, string b) {return a.size() < b.size();})->size();
	
	cout << indent(1) << "Confusion Matrix:" << endl;
	
	cout << setw(largestClassName + 4) << " ";
	for_each(classes.begin(), classes.end(), [](string a){
		cout << " " << a.substr(0, 4);
	});
	cout << endl;
	
	float diagonalTotal = 0;
	for(unsigned int i = 0; i < classes.size(); i++) {
		cout << indent(2) << setw(largestClassName) << classes[i];
		for(unsigned int j = 0; j < classes.size(); j++) {
			float percentage = confusionMatrix[i][j] * 100.0;
			cout << " " << setiosflags(ios::fixed) <<
				setprecision(1) << setw(4) << percentage;
			
			if(i == j) {
				diagonalTotal += percentage;
			}
		}
		cout << endl;
	}
	float diagonalAvg = diagonalTotal / classes.size();
	cout << indent(2) << "Diagonal: " << diagonalAvg << "%" << endl;
}

string OutputHelper::clearLine() {
	stringstream ss;
	ss << "\r" << setw(80) << " " << "\r";
	return ss.str();
}

string OutputHelper::indent(int level) {
	if(level == 0) return "";
	stringstream ss;
	ss << setw(2 * level) << " ";
	return ss.str();
}
