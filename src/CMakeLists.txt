cmake_minimum_required(VERSION 2.8)
SET(EXECUTABLE_OUTPUT_PATH ../${PROJECT_BINARY_DIR})
project(DetectingNature)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_executable(DetectingNature main.cpp)

target_link_libraries(DetectingNature ${OpenCV_LIBS})
target_link_libraries(DetectingNature ${Boost_LIBRARIES})

