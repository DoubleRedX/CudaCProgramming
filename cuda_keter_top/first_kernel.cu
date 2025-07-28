//
// Created by Gary27 on 2025/7/7.
//

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

template<typename T, size_t N>
void initialize_array_on_host(T *x) {
    for (int i=0;i<N;i++) x[i] = i;
}

// template<typename T, size_t N>
// __global__ void vec_add(T *a, T *b)

int main() {

    const uint32_t N = 1024;
    auto* x_host = new float[N];
    initialize_array_on_host<float, N>(x_host);
    float *x_cuda;
    auto bytes = sizeof(float) * N;
    if (cudaMalloc(&x_cuda, bytes) != cudaSuccess) { std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl; };
    if (cudaMemcpy(x_cuda, x_host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) { std::cout << cudaGetLastError() << std::endl; };






    return 0;
}