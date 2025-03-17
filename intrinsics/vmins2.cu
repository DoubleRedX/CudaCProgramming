//
// Created by Gary27 on 2025/3/17.
//
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_vmins2() {
    short2 a = {10, 20};
    short2 b = {5, 25};
    unsigned int r_a = *reinterpret_cast<unsigned int*>(&a);
    unsigned int r_b = *reinterpret_cast<unsigned int*>(&b);
    unsigned int ret = __vmins2(r_a, r_b);
    short2 result = *reinterpret_cast<short2*>(&ret);

    printf("Result of __vmins2: (%d, %d)\n", result.x, result.y);
}

int main() {
    test_vmins2<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}