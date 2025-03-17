//
// Created by Gary27 on 2025/3/17.
//
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_vmins4() {
    char4 a = {10, 20, 127, 45};
    char4 b = {5, 25, 30, 55};
    unsigned int r_a = *reinterpret_cast<unsigned int*>(&a);
    unsigned int r_b = *reinterpret_cast<unsigned int*>(&b);
    unsigned int ret = __vmins4(r_a, r_b);
    char4 result = *reinterpret_cast<char4*>(&ret);

    printf("Result of __vmins4: (%d, %d, %d, %d)\n", result.x, result.y, result.z, result.w);
}

int main() {
    test_vmins4<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}