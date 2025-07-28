//
// Created by Gary27 on 2025/7/3.
//

#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceID = 0;
    if (cudaSuccess != cudaGetDevice(&deviceID)) return -10;

    cudaSharedMemConfig sharedMemConfig;
    cudaDeviceGetSharedMemConfig(&sharedMemConfig);
    std::cout << "cudaSharedMemConfig: " << sharedMemConfig << std::endl;

}