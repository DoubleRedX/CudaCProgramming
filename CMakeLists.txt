cmake_minimum_required(VERSION 3.16)
project(CudaCProgramming LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(CudaCProgramming main.cu)
set_target_properties(CudaCProgramming PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sumArrayOnHost sumArrayOnHost.cu)
#set_target_properties(CudaCProgramming PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(sumArrayOnDevice sumArrayOnDevice.cu)
add_executable(checkDimension checkDimension.cu)
add_executable(defineGridBlock defineGridBlock.cu)
add_executable(sumArraysOnGPU-small-case sumArraysOnGPU-small-case.cu)
add_executable(malloc malloc.cpp)
add_executable(sumArraysOnGPUwithTime sumArraysOnGPUwithTime.cu)
add_executable(checkThreadIndex checkThreadIndex.cu)
add_executable(sumMatrixOnGPU-2d-grid-2d-block sumMatrixOnGPU-2d-grid-2d-block.cu)