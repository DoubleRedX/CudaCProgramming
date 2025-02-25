//
// Created by Gary27 on 2025/2/25.
//
#include <cuda_runtime.h>
#include <iostream>

__global__ void copyKernel(float* in, float* out){
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    out[idx] = in[idx];
}

__global__ void copyKernel2(float* in, float* out){
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    out[idx] = __ldg(&in[idx]);
}

__global__ void copyKernel3(float* __restrict__ in, float* __restrict__ out){
    auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    out[idx] = in[idx];
}


int main(int argc, char** argv){

    int dev_idx = 0;
    cudaSetDevice(dev_idx);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev_idx);
    std::cout << "Device name: " << prop.name << "\n";

    int data_size = 1 << 22;
    int n_bytes = data_size * sizeof(data_size);


    return 0;
}