cmake_minimum_required(VERSION 2.8)
project(DetectingNature)
cmake_policy(SET CMP0015 NEW)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem serialization program_options REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/..)
set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra -Wno-unused-parameter -std=c++11 ${OpenMP_CXX_FLAGS}")

include_directories(.)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(../libs/vlfeat-0.9.16/)
link_directories(../libs/vlfeat-0.9.16/bin/glnx86/)

add_executable(DetectingNature
	utils/OutputHelper.cpp
	utils/CacheHelper.cpp
	features/Image.cpp
	features/ImageFeatures.cpp
	features/FeatureExtractor.cpp
	framework/Settings.cpp
	utils/DatasetManager.cpp
	codebook/Histogram.cpp
	codebook/Codebook.cpp
	codebook/CodebookGenerator.cpp
	classification/ConfusionMatrix.cpp
	classification/Classifier.cpp
	framework/ClassificationFramework.cpp
	main.cpp)

target_link_libraries(DetectingNature ${OpenCV_LIBS})
target_link_libraries(DetectingNature ${Boost_LIBRARIES})
target_link_libraries(DetectingNature vl)
target_link_libraries(DetectingNature svm)

install(TARGETS DetectingNature DESTINATION bin)
