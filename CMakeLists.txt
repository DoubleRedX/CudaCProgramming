cmake_minimum_required(VERSION 3.16)

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES 75)
#endif()
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")

project(CudaCProgramming LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_subdirectory(c01)

##set_target_properties(CudaCProgramming PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#
#add_executable(sumArrayOnHost sumArrayOnHost.cu)
##set_target_properties(CudaCProgramming PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#
#add_executable(sumArrayOnDevice sumArrayOnDevice.cu)
#add_executable(checkDimension checkDimension.cu)
#add_executable(defineGridBlock defineGridBlock.cu)
#add_executable(sumArraysOnGPU-small-case sumArraysOnGPU-small-case.cu)
#add_executable(malloc malloc.cpp)
#add_executable(sumArraysOnGPUwithTime sumArraysOnGPUwithTime.cu)
#add_executable(checkThreadIndex checkThreadIndex.cu)
#add_executable(sumMatrixOnGPU-2d-grid-2d-block sumMatrixOnGPU-2d-grid-2d-block.cu)
#
#
#add_executable(2d_grid 2d_grid.cu)