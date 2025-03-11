//
// Created by Gary27 on 2025/3/7.
//

#include <cuda_runtime.h>

#include <cstdio>

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


int main(int argc, char** argv){


    const int originalHeight = 3;
    const int originalWidth = 4;

    const int size = originalWidth * originalHeight * sizeof(float);

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

    float h_output[originalWidth * originalHeight];

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize(
            (originalWidth + blockSize.x - 1) / blockSize.x,
            (originalHeight + blockSize.y - 1) / blockSize.y
    );
    matrix_transpose_row<<<gridSize, blockSize>>>(d_input, d_output, originalHeight, originalWidth);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < originalWidth; ++i) {
        for (int j = 0; j < originalHeight; ++j) {
            printf("%.0f ", h_output[i * originalHeight + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);







    return 0;
}