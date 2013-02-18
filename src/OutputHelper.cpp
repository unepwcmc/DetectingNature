#include "OutputHelper.h"
using namespace std;
using namespace boost;

OutputHelper::OutputHelper() {
}

void OutputHelper::printInlineMessage(string message,
		unsigned int indentLevel) {
		
	cout << clearLine() << indent(indentLevel) << message;
	cout.flush();
}

void OutputHelper::printMessage(string message, unsigned int indentLevel) {
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
	cout.flush();
}

void OutputHelper::printResults(unsigned int correctTrain, unsigned int sizeTrain,
		 unsigned int correctTest, unsigned int sizeTest) {
		 
	int percentRecall = correctTrain * 100.0 / sizeTrain;	
	
	int percentRecognition = correctTest * 100.0 / sizeTest;
		
	cout << clearLine() << indent(2) << "Recall: "
		<< percentRecall << "%" << endl;
	cout << indent(2) << "Recognition: " << percentRecognition << "%" << endl;
}

void OutputHelper::printConfusionMatrix(const vector<string>& classes,
		const multi_array<float, 2>& confusionMatrix) {
	
	int largestClassName = max_element(classes.begin(), classes.end(),
		[](string a, string b) {return a.size() < b.size();})->size();
	
	cout << indent(1) << "Confusion Matrix:" << endl;
	
	cout << setw(largestClassName + 4) << " ";
	for_each(classes.begin(), classes.end(), [](string a){
		cout << " " << a.substr(0, 2);
	});
	cout << endl;
	
	unsigned int diagonalTotal = 0;
	for(unsigned int i = 0; i < classes.size(); i++) {
		cout << indent(2) << setw(largestClassName) << classes[i];
		for(unsigned int j = 0; j < classes.size(); j++) {
			int percentage = confusionMatrix[i][j] * 100;
			cout << " " << setw(2) << percentage;
			
			if(i == j) {
				diagonalTotal += percentage;
			}
		}
		cout << endl;
	}
	int diagonalAvg = diagonalTotal / classes.size();
	cout << indent(2) << "Diagonal: " << diagonalAvg << "%" << endl;
}

string OutputHelper::clearLine() {
	stringstream ss;
	ss << "\r" << setw(80) << " " << "\r";
	return ss.str();
}

string OutputHelper::indent(int level) {
	stringstream ss;
	ss << setw(2 * level) << " ";
	return ss.str();
}
