cmake_minimum_required(VERSION 2.8)
project(DetectingNature)

# -----------------------------------------------------------------------------
# Set CMAKE parameters
# -----------------------------------------------------------------------------

cmake_policy(SET CMP0015 NEW)
set(CMAKE_INSTALL_RPATH ".")
set(CMAKE_CXX_FLAGS "-g -O2 -march=native -Wall -Wextra -Wno-unused-parameter -std=c++11")

# -----------------------------------------------------------------------------
# Locate required libraries and components
# -----------------------------------------------------------------------------

find_package(Boost 1.50.0 COMPONENTS system filesystem serialization program_options REQUIRED)

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(SWIG 2.0.0)
if(SWIG_FOUND)
	find_package(Ruby 1.9.1 REQUIRED)
	include(${SWIG_USE_FILE})
	set_source_files_properties(DetectingNature.i PROPERTIES CPLUSPLUS ON)
endif()

# -----------------------------------------------------------------------------
# Define include files and library directories
# XXX Fix these horrible relative paths
# -----------------------------------------------------------------------------

include_directories(
	.
	${Boost_INCLUDE_DIRS}
	../libs/vlfeat-0.9.16/
	../libs/CImg-1.5.4/
	../libs/yael_v300/
	../libs/gmm-fisher/
)

if(SWIG_FOUND)
	include_directories(
		${RUBY_INCLUDE_DIRS}
	)
endif()

link_directories(
	../libs/vlfeat-0.9.16/bin/glnx86/
	../libs/yael_v300/yael/
	../libs/gmm-fisher/
)

install(FILES ../libs/vlfeat-0.9.16/bin/glnx86/libvl.so DESTINATION .)
install(FILES ../libs/yael_v300/yael/libyael.so DESTINATION .)
install(FILES ../libs/gmm-fisher/libgmmfisher.so DESTINATION .)

# -----------------------------------------------------------------------------
# Define this project's files and dependencies
# -----------------------------------------------------------------------------

set(DETECTINGNATURE_SOURCE_FILES
	utils/OutputHelper.cpp
	utils/CacheHelper.cpp
	images/ImageData.cpp
	images/ImageLoader.cpp
	images/HSVImageLoader.cpp
	images/OpponentImageLoader.cpp
	images/GreyscaleImageLoader.cpp
	features/ImageFeatures.cpp
	features/LBPFeatureExtractor.cpp
	features/HOGFeatureExtractor.cpp
	features/SIFTFeatureExtractor.cpp
	features/HellingerFeatureTransform.cpp
	utils/DatasetManager.cpp
	codebook/Histogram.cpp
	codebook/KMeansCodebook.cpp
	codebook/FisherCodebook.cpp
	codebook/CodebookGenerator.cpp
	codebook/KMeansCodebookGenerator.cpp
	codebook/FisherCodebookGenerator.cpp
	classification/ConfusionMatrix.cpp
	classification/SVMClassifier.cpp
	classification/LinearClassifier.cpp
	framework/ClassificationFramework.cpp
)

set(DETECTINGNATURE_LIBRARIES
	${Boost_LIBRARIES}
	vl
	linear
	svm
	yael
	gmmfisher
)

# -----------------------------------------------------------------------------
# Build the C++ library
# -----------------------------------------------------------------------------

add_library(detectingnature
	${DETECTINGNATURE_SOURCE_FILES}
)

target_link_libraries(detectingnature
	${DETECTINGNATURE_LIBRARIES}
)

# -----------------------------------------------------------------------------
# Build the C++ command line app
# -----------------------------------------------------------------------------

add_executable(DetectingNatureDemo
	main.cpp
)

target_link_libraries(DetectingNatureDemo
	detectingnature
)


set_target_properties(DetectingNatureDemo PROPERTIES OUTPUT_NAME DetectingNature)
install(TARGETS DetectingNatureDemo DESTINATION .)

# -----------------------------------------------------------------------------
# Build the ruby wrapper
# -----------------------------------------------------------------------------

if(SWIG_FOUND)
	swig_add_module(DetectingNatureRuby ruby
		DetectingNature.i
	)

	swig_link_libraries(DetectingNatureRuby
		detectingnature
		${RUBY_LIBRARIES}
	)

	set_target_properties(DetectingNatureRuby
		PROPERTIES OUTPUT_NAME DetectingNature PREFIX "")
	install(TARGETS DetectingNatureRuby DESTINATION .)
	install(FILES RubyDemo.rb DESTINATION .)
endif()
