#include <cstdio>
#include <iostream>
#include <chrono>
#include "helper.h"

constexpr int M = 128;
constexpr int N = 256;


__global__ void idx_check(
        float A[M][N]
) {
    int i = threadIdx.x;
    printf("A");
}

int main() {

    dim3 BlockSize(16, 32);
    dim3 numBlocks(M / BlockSize.x, N / BlockSize.y);
    float A[M][N]{};
    idx_check<<<numBlocks, BlockSize>>>(A);

//    std::chrono::

//    cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}