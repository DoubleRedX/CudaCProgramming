//
// Created by Gary27 on 2025/3/17.
//

#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_vabsss2() {
    short2 u = {-10, 20};
    unsigned int r_u = *reinterpret_cast<unsigned int *>(&u);
    unsigned int res = __vabsss2(r_u);
    short2 result = *reinterpret_cast<short2 *>(&res);
    printf("Result of __vabsss2: (%d, %d)\n", result.x, result.y);
}

int main() {
    test_vabsss2<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
