#include "ClassificationFramework.h"

int main(int argc, char** argv) {
	Settings settings("settings.xml");
	if(argc == 2) {
		settings = Settings(std::string(argv[1]));
	}
	
	ClassificationFramework cf(settings);
	cf.run();
	
	return 0;
}
