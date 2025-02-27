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
    int n_bytes = data_size * sizeof(float);

    auto *hst_mem_A = new float[data_size];
    auto *hst_mem_B = new float[data_size];

    std::fill_n(hst_mem_A, data_size, 1.0);
    std::fill_n(hst_mem_B, data_size, 2.0);

    float *dvc_mem_A, *dvc_mem_B;

    cudaMalloc(&dvc_mem_A, n_bytes);
    cudaMalloc(&dvc_mem_B, n_bytes);
    cudaMemcpy(dvc_mem_A, hst_mem_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_mem_B, hst_mem_B, n_bytes, cudaMemcpyHostToDevice);

    dim3 bs {32, 1, 1};
    dim3 grid {(bs.x + data_size + 1) / bs.x, 1, 1};

    copyKernel<<<grid, bs>>>(dvc_mem_B, dvc_mem_A);
    copyKernel2<<<grid, bs>>>(dvc_mem_A, dvc_mem_B);
    copyKernel3<<<grid, bs>>>(dvc_mem_B, dvc_mem_A);

    delete[] hst_mem_B;
    delete[] hst_mem_A;

    cudaFree(dvc_mem_A);
    cudaFree(dvc_mem_B);

    cudaDeviceReset();
    return 0;
}

// 为什么profile不出来memory的信息？