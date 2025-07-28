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

__device__ void ldg(){
    uchar2 uc2 {50, 100};
    const uchar2* uc2_ptr = &uc2;
    auto res = __ldg(uc2_ptr);
    printf("%d, %d", res.x, res.y);
}

__global__ void device_call(){
    uchar2 uc2 {50, 100};
    const uchar2* uc2_ptr = &uc2;
    auto res = __ldg(uc2_ptr);
    printf("222  %d, %d", res.x, res.y);
}


int main() {
    test_vabsss2<<<1, 1>>>();
    cudaDeviceSynchronize();

    device_call<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
