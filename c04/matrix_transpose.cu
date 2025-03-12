//
// Created by Gary27 on 2025/3/7.
//

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <iostream>
#include "helper.h"

template<typename T>
__global__ void matrix_transpose_row(T *in, T *out, int m, int n){
    // 按行读取：threadIdx.x维度所在为矩阵的行
    auto i = threadIdx.y + blockIdx.y * blockDim.y;  // 两个kernel中的i/j都遵循数学中i和j的定义。
    auto j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < m && j < n){
        out[i + j * m] = in[j + i * n];  // read i, j --> write j, i
    }
}

template<typename T>
__global__ void matrix_transpose_col(T *in, T *out, int m, int n){
    // 按列读取：threadIdx.x维度所在为矩阵的列
    auto i = threadIdx.x + blockDim.x * blockIdx.x;
    auto j = threadIdx.y + blockDim.y * blockIdx.y;
    if(i < m && j < n){
        out[i + j * m] = in[j + i * n];
    }
}

template<typename T>
__global__ void matrix_transpose_row_unroll4(T *in, T *out, int m, int n){
    // 按列读取：threadIdx.x维度所在为矩阵的列
    auto j = threadIdx.x + blockDim.x * blockIdx.x * 4;
    auto i = threadIdx.y + blockDim.y * blockIdx.y;

    if (i >= m) return;

    auto read_idx = j + i * m;
    auto write_idx = i + j * n;
    if(i < m && (j + blockDim.x * 3) < n){
        out[write_idx] = in[read_idx];
        out[write_idx + m * blockDim.x] = in[read_idx + blockDim.x];
        out[write_idx + m * blockDim.x * 2] = in[read_idx + blockDim.x * 2];
        out[write_idx + m * blockDim.x * 3] = in[read_idx + blockDim.x * 3];
    }
}

template<typename T>
__global__ void matrix_transpose_row_unroll(T *in, T *out, int ny, int nx){
    // 按列读取：threadIdx.x维度所在为矩阵的列
    auto ix = threadIdx.x + blockDim.x * blockIdx.x * 4;
    auto iy = threadIdx.y + blockDim.y * blockIdx.y;
    auto ti = ix + iy * nx;
    auto to = iy + ix * ny;
    if(iy < ny && (ix + blockDim.x * 3) < nx){
        out[to] = in[ti];
        out[to + ny * blockDim.x] = in[ti + blockDim.x];
        out[to + ny * blockDim.x * 2] = in[ti + blockDim.x * 2];
        out[to + ny * blockDim.x * 3] = in[ti + blockDim.x * 3];
    }
}


int main(int argc, char** argv){


    const int originalHeight = 64;
    const int originalWidth = 64;
    const long long size = originalWidth * originalHeight * sizeof(float);

    auto *h_input = new float[originalHeight * originalWidth];
    for (int i=0; i < originalHeight; ++i){
        for (int j=0;j < originalWidth; ++j){
            h_input[j + i * originalWidth] = float (j + i * originalWidth);
        }
    }

    for (int i = 0; i < originalHeight; ++i) {
        for (int j = 0; j < originalWidth; ++j) {
            printf("%.0f ", h_input[i * originalWidth + j]);
        }
        printf("\n");
    }

//    float h_output[originalWidth * originalHeight];  // 栈上分配
    auto *h_output = new float[originalWidth * originalHeight];

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32);
    dim3 gridSize(
            (originalWidth + blockSize.x - 1) / blockSize.x,
            (originalHeight + blockSize.y - 1) / blockSize.y
    );
    auto start = std::chrono::high_resolution_clock::now();
    matrix_transpose_row_unroll4<<<gridSize, blockSize>>>(d_input, d_output, originalHeight, originalWidth);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;  // second

    float ibnd = originalHeight * originalWidth * 2 * sizeof(float) / 1e9 / duration.count();
    std::cout << "Kern exec time: " << duration.count() << "s" << std::endl;
    std::cout << "Effective Bandwidth: " << ibnd << "GB/s" << std::endl;


    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < originalWidth; ++i) {
        for (int j = 0; j < originalHeight; ++j) {
            printf("%.0f ", h_output[i * originalHeight + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    return 0;
}