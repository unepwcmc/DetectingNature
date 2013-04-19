cmake_minimum_required(VERSION 2.8)
project(gmmfisher)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(gmmfisher SHARED
	gmm.cxx
	simd_math.cxx
	stat.cxx
	fisher.cxx)
