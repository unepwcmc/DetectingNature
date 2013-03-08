#include "ClassificationFramework.h"

int main() {
	Settings settings;
	settings.datasetPath = "data/OT-2";
	
	ClassificationFramework cf(settings);
	cf.run();
	
	return 0;
}
