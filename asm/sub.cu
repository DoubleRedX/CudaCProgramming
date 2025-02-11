//
// Created by Gary27 on 2025/1/20.
//
#include <cstdio>

__global__ void subtract_kernel(unsigned int *result, unsigned int a, unsigned int b) {
    asm volatile (
            "sub.u32 %0, %1, %2;"  // 执行减法操作：result = a - b  // %0 = %1 - %2
            : "=r"(*result)        // 输出操作数，结果存储在 result 中
            : "r"(a), "r"(b)       // 输入操作数，a 和 b 作为源操作数
            );
}

int main() {
    unsigned int h_result = 0;
    unsigned int *d_result;
    unsigned int a = 10;
    unsigned int b = 3;

    // 分配设备内存
    cudaMalloc((void**)&d_result, sizeof(unsigned int));

    // 调用 CUDA 核函数
    subtract_kernel<<<1, 1>>>(d_result, a, b);

    // 将结果从设备复制到主机
    cudaMemcpy(&h_result, d_result, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // 输出结果
    printf("Result of %u - %u = %u\n", a, b, h_result);

    // 释放设备内存
    cudaFree(d_result);

    return 0;
}