#include "ClassificationFramework.h"

int main() {
	Settings settings;
	settings.datasetPath = "data/OT-8";
	settings.colourspace = Image::HSV;
	
	ClassificationFramework cf(settings);
	cf.run();
	
	return 0;
}
