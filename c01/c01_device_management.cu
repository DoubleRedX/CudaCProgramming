//
// Created by Gary27 on 2024/12/30.
//

#include <cuda_runtime.h>
#include <iostream>

void coutNvidiaDeviceInfo(int deviceCnt = 0) {
  if (deviceCnt == 0) {
    return;
  }
  cudaDeviceProp deviceProp;
  int cudaDriverVersion = 0;
  int cudaRuntimeVersion = 0;
  for (int i = 0; i < deviceCnt; ++i) {
    cudaGetDeviceProperties(&deviceProp, i);
    cudaDriverGetVersion(&cudaDriverVersion);
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
    std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
    std::cout << "CUDA Driver Version: " << cudaDriverVersion / 1000 << "." << (cudaDriverVersion % 1000) / 10 << std::endl;
    std::cout << "CUDA Runtime Version: " << cudaRuntimeVersion / 1000 << "." << (cudaRuntimeVersion % 1000) / 10 << std::endl;
    std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Maximum Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Maximum Thread Dimensions: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "  Maximum Grid Size: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
    std::cout << "  Clock Rate: " << deviceProp.clockRate << " MHz" << std::endl;
    std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / (1024 * 1024) << " KB" << std::endl;
    std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << std::endl;
  }
}

int main(){

  int deviceCnt = 0;
  auto cuError = cudaGetDeviceCount(&deviceCnt);
  if (cuError != cudaSuccess) {
    std::cout << "cudaGetDeviceCount error: " << cudaGetErrorString(cuError) << std::endl;
    return -1;
  }
  std::cout << "Number of NVIDIA GPUs available: " << deviceCnt << std::endl;
  coutNvidiaDeviceInfo(deviceCnt);
  return 0;
}