cmake_minimum_required(VERSION 2.8)
project(DetectingNature)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/..)
set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra -std=c++11 ${OpenMP_CXX_FLAGS}")

include_directories(${Boost_INCLUDE_DIRS})

add_executable(DetectingNature DatasetManager.cpp OutputHelper.cpp TrainingSettings.cpp CodebookManager.cpp main.cpp)

target_link_libraries(DetectingNature ${OpenCV_LIBS})
target_link_libraries(DetectingNature ${Boost_LIBRARIES})

