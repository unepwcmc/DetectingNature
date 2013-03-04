#include "ClassificationFramework.h"

int main() {
	ClassificationFramework::Settings settings;
	settings.datasetPath = "data/scene_categories";
	
	ClassificationFramework cf(settings);
	cf.run();
	
	return 0;
}
