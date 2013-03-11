cmake_minimum_required(VERSION 2.8)
project(DetectingNature)
cmake_policy(SET CMP0015 NEW)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem serialization REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/..)
set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra -Wno-unused-parameter -std=c++11 ${OpenMP_CXX_FLAGS}")

include_directories(${Boost_INCLUDE_DIRS})
include_directories(../libs/vlfeat-0.9.16)
link_directories(../libs/vlfeat-0.9.16/bin/glnx86/)

add_executable(DetectingNature OutputHelper.cpp CacheHelper.cpp Image.cpp ImageFeatures.cpp FeatureExtractor.cpp Settings.cpp DatasetManager.cpp Histogram.cpp Codebook.cpp CodebookGenerator.cpp ConfusionMatrix.cpp Classifier.cpp ClassificationFramework.cpp main.cpp)

target_link_libraries(DetectingNature ${OpenCV_LIBS})
target_link_libraries(DetectingNature ${Boost_LIBRARIES})
target_link_libraries(DetectingNature vl)
target_link_libraries(DetectingNature svm)

install(TARGETS DetectingNature DESTINATION bin)
