%module DetectingNature

#pragma SWIG nowarn=SWIGWARN_PARSE_NESTED_CLASS

%include <std_string.i>
%include <std_vector.i>
%include <exception.i>

%exception {
	try {
		$action
	} catch(const std::exception& e) {
		SWIG_exception(SWIG_RuntimeError, e.what());
	}
}


%{
#include "framework/ClassificationFramework.h"
%}


%nestedworkaround ClassificationFramework::Result;

struct Result {
	std::string filepath;
	std::string category;
	double certainty;
};

%include "framework/ClassificationFramework.h"

%{
typedef ClassificationFramework::Result Result;
%}

namespace std {
	%template(Results) vector<ClassificationFramework::Result>;
};

%{
#include "framework/SettingsManager.h"
%}
%include "framework/SettingsManager.h"
