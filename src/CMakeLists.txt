cmake_minimum_required(VERSION 2.8)
SET(EXECUTABLE_OUTPUT_PATH ../${PROJECT_BINARY_DIR})
project(DetectingNature)

find_package(OpenCV REQUIRED)
add_executable(DetectingNature main.cpp)
target_link_libraries(DetectingNature ${OpenCV_LIBS})


