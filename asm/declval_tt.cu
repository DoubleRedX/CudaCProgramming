//
// Created by Gary27 on 2025/3/20.
//
#include <cuda_runtime.h>
#include <cstdio>

__device__ void ldg(){
    uchar2 uc2 {50, 100};
    const uchar2* uc2_ptr = &uc2;
    auto res = __ldg(uc2_ptr);
    printf("%u, %u", res.x, res.y);
}

__global__ void device_call(){
    ldg();
}

int main() {

    // 调用设备代码
    device_call<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}