//
// Created by Gary27 on 2025/2/11.
//

#include <cuda_runtime.h>
#include <cstdio>
#include "spdlog/spdlog.h"

__device__ float devData = 1.0f;

__global__ void checkGlobalVariable(){
    printf("Device: devData is %f\n", devData);
    devData += 2.0f;
}

int main(){
    float value = 10.;
    cudaMemcpyToSymbol(devData, &value, sizeof(float ));
    checkGlobalVariable<<<1,1>>>();
    // ---------------核心代码------------------------
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    // ---------------核心代码------------------------
//    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&value, devData, sizeof(float ));
    printf("Host: value is %f\n", value);
    printf("Host: devData is %f\n", devData);
//    spdlog::info("Host: Value is {}\n", value);
    cudaDeviceReset();
    return 0;
}