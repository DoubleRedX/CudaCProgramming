cmake_minimum_required(VERSION 3.16)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 86)

project(CudaCProgramming LANGUAGES C CXX CUDA)

include(cmake/CPM.cmake)

#CPMAddPackage(
#        NAME spdlog
##        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.0.zip
#        SOURCE_DIR /data3/cll/codes/cprojects/book_cuda_c_programming/3rd/spdlog-1.15.1
#)  # EXCLUDE_FROM_ALL？

if(spdlog_ADDED)
    link_libraries(spdlog::spdlog)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86)

# 查找CUDA并检测架构
find_package(CUDAToolkit REQUIRED)
message(CMAKE_CUDA_ARCHITECTURES:  ${CMAKE_CUDA_ARCHITECTURES})
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")  # 动态监测GPU架构
    execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_ARCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (GPU_ARCH)
        # 提取计算能力并设置 CMAKE_CUDA_ARCHITECTURES
        # 多GPU系统 和 单GPU系统统一考虑
        #        string(REPLACE "\n" ";" GPU_ARCH_LIST "${GPU_ARCH}")
        #        list(GET GPU_ARCH_LIST 0 GPU_ARCH)  # 选择第一个 GPU 的计算能力

        string(REPLACE "." "" GPU_ARCH "${GPU_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH} CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Detected GPU architecture: sm_${GPU_ARCH}")
    else()
        # 如果检测失败，报致命错误
        message(FATAL_ERROR "Failed to detect GPU architecture. Please manually set CMAKE_CUDA_ARCHITECTURES.")
    endif()
endif()


include_directories(include)
add_subdirectory(c01)
add_subdirectory(c03)
add_subdirectory(c04)
add_subdirectory(c05)

add_subdirectory(asm)
add_subdirectory(intrinsics)
add_subdirectory(nvidia_cuda_api)
add_subdirectory(quantization)

add_subdirectory(utils)
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

add_subdirectory(meta)
add_subdirectory(cpp)


add_subdirectory(share_mem)
add_subdirectory(cuda_keter_top)
add_subdirectory(cooperative)
add_subdirectory(leetcuda)