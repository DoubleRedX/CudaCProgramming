cmake_minimum_required(VERSION 3.16)
project(CudaCProgramming LANGUAGES C CXX CUDA)

include(cmake/CPM.cmake)

CPMAddPackage(
        NAME spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.0.zip
)

if(spdlog_ADDED)
    link_libraries(spdlog::spdlog)
endif()

include_directories(include)
add_subdirectory(c01)
add_subdirectory(c03)
add_subdirectory(c04)

add_subdirectory(asm)
add_subdirectory(intrinsics)
add_subdirectory(nvidia_cuda_api)
add_subdirectory(quantization)
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

add_executable(main main.cpp)