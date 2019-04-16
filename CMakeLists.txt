cmake_minimum_required(VERSION 3.0)

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  	set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
	  CACHE STRING "")
	MESSAGE(STATUS "Using vcpkg")
endif()

project(traph)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

SET(TRAPH_PATH ${PROJECT_SOURCE_DIR} CACHE STRING "Adds a path to TRAPH" FORCE)
SET(TRAPH_PATH_INCLUDE ${TRAPH_PATH}/traph/include CACHE STRING "Adds a path to TRAPH header" FORCE)
SET(TRAPH_PATH_HEADER ${TRAPH_PATH_INCLUDE}/traph CACHE STRING "Adds a path to TRAPH header" FORCE)
SET(TRAPH_PATH_SOURCE ${TRAPH_PATH}/traph/source CACHE STRING "Adds a path to TRAPH source" FORCE)
SET(TRAPH_PATH_DEPENDENCIES ${TRAPH_PATH}/traph/contrib CACHE STRING "Adds a path to TRAPH dependencies" FORCE)

find_package(Boost)
if(Boost_FOUND)
	message(STATUS Boost found: ${Boost_INCLUDE_DIRS})
endif()


# languge feature
set(ENABLE_PYTHON True CACHE BOOL "Language: Python")

# gpgpu feature
set(ENABLE_CUDA FALSE CACHE BOOL "Feature: OPENCL")
set(ENABLE_OPENCL FALSE CACHE BOOL "Feature: OPENCL")

IF(ENABLE_CUDA)
	SET(TRAPH_ENABLE_CUDA TRUE)
	ADD_DEFINITIONS(-DTRAPH_ENABLE_CUDA)

	FIND_PACKAGE(CUDA REQUIRED)
	SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
	IF(CUDA_FOUND)
		include_directories(${CUDA_INCLUDE_DIRS})
		message(STATUS cuda found)
	ENDIF()
ENDIF()

IF(ENABLE_OPENCL)
	SET(TRAPH_ENABLE_OPENCL TRUE)
	ADD_DEFINITIONS(-DTRAPH_ENABLE_OPENCL)

	find_package(OpenCL REQUIRED)
	if (OpenCL_FOUND)
		include_directories(${OpenCL_INCLUDE_DIRS})
		message(STATUS opencl found: ${OpenCL_INCLUDE_DIRS})
	endif()
ENDIF()

# blas feature
SET(TRAPH_ACCELERATE 0 CACHE STRING
"Specify the feature Possible values:
0 - only eigen
1 - mkl
2 - openblas")

IF(TRAPH_ACCELERATE EQUAL 0)
	SET(TRAPH_BUILD_EIGEN TRUE)
	ADD_DEFINITIONS(-DTRAPH_BUILD_EIGEN)

	message(STATUS using eigen blas)
ELSEIF(TRAPH_ACCELERATE EQUAL 1)
	SET(TRAPH_BUILD_MKL TRUE)
	ADD_DEFINITIONS(-DTRAPH_BUILD_MKL)

	# set(BLA_VENDOR Intel)
	find_package(BLAS REQUIRED)
	if (BLAS_FOUND)
		include_directories(${BLAS_INCDIR})
		message(STATUS BLAS found: ${BLAS_DIR})
	endif()
ELSEIF(TRAPH_ACCELERATE EQUAL 2)
	SET(TRAPH_BUILD_OPENBLAS TRUE)
	ADD_DEFINITIONS(-DTRAPH_BUILD_OPENBLAS)

	set(BLA_VENDOR Open)
	find_package(BLAS REQUIRED)
	if (BLAS_FOUND)
		include_directories(${BLAS_INCDIR})
		message(STATUS BLAS found: ${BLAS_DIR})
	endif()
ELSE()
	MESSAGE(FATAL_ERROR "Unsupported build platform: " ${OCTOON_BUILD_PLATFORM})
ENDIF()

ADD_SUBDIRECTORY(traph)
