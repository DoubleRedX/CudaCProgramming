//
// Created by Gary27 on 2025/3/7.
//

#include <cuda_runtime.h>

#include <cstdio>

template<typename T>
__global__ void matrix_transpose_row(T *in, T *out, int m, int n){
    auto i = threadIdx.y + blockIdx.y * blockDim.y;
    auto j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < m && j < n){
        out[i + j * m] = in[j + i * n];  // read i, j --> write j, i
    }
}
//
//template<typename T>
//__global__ void matrix_transpose_col(T *in, T *out, int m, int n){
//    auto i = threadIdx.y + blockDim.y * blockIdx.y;
//    auto j = threadIdx.x + blockDim.x * blockIdx.x;
//    if(i < m && j < n){
//        out[] = in[];
//    }
//}


int main(int argc, char** argv){


    const int originalHeight = 3;
    const int originalWidth = 4;

    const int size = originalWidth * originalHeight * sizeof(float);

    float h_input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                       5.0f, 6.0f, 7.0f, 8.0f,
                       9.0f, 10.0f, 11.0f, 12.0f};
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
            printf("%.0f ", h_output[i * originalWidth + j]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);







    return 0;
}