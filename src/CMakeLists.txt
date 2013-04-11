cmake_minimum_required(VERSION 2.8)
project(DetectingNature)
cmake_policy(SET CMP0015 NEW)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/..)
set(CMAKE_CXX_FLAGS "-g -O2 -march=native -Wall -Wextra -Wno-unused-parameter -std=c++11")

find_package(OpenMP REQUIRED)
find_package(Boost COMPONENTS system filesystem serialization program_options REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
	.
	${Boost_INCLUDE_DIRS}
	../libs/vlfeat-0.9.16/
	../libs/CImg-1.5.4/
	../libs/yael_v300/)
	
link_directories(
	../libs/vlfeat-0.9.16/bin/glnx86/
	../libs/yael_v300/yael/)

add_executable(DetectingNature
	utils/OutputHelper.cpp
	utils/CacheHelper.cpp
	features/Image.cpp
	features/ImageFeatures.cpp
	features/LBPFeatureExtractor.cpp
	features/HOGFeatureExtractor.cpp
	features/SIFTFeatureExtractor.cpp
	features/HellingerFeatureTransform.cpp
	utils/DatasetManager.cpp
	codebook/Histogram.cpp
	codebook/Codebook.cpp
	codebook/CodebookGenerator.cpp
	classification/ConfusionMatrix.cpp
	classification/Classifier.cpp
	framework/ClassificationFramework.cpp
	main.cpp)

target_link_libraries(DetectingNature
	${Boost_LIBRARIES}
	vl
	linear
	svm
	yael)

install(TARGETS DetectingNature DESTINATION bin)
