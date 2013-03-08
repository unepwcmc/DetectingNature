#include "ClassificationFramework.h"

int main() {
	ClassificationFramework::Settings settings;
	settings.datasetPath = "data/OT-2";
	
	ClassificationFramework cf(settings);
	cf.run();
	
	return 0;
}
