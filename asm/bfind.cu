//
// Created by Gary27 on 2025/1/16.
//


#include <cuda_runtime.h>
#include <iostream>

__device__ unsigned int find_highest_bit(unsigned int value) {
    unsigned int result;
    asm("bfind.u32 %0, %1;" : "=r"(result) : "r"(value)); // 内联汇编调用 bfind.u32
    return result;
}

__global__ void test_kernel(unsigned int* output, unsigned int input) {
    output[0] = find_highest_bit(input); // 调用函数查找最高有效位
}

int main() {
    unsigned int h_input = 0x00080000; // 输入值
    unsigned int* d_output;
    cudaMalloc(&d_output, sizeof(unsigned int)); // 分配设备内存

    test_kernel<<<1, 1>>>(d_output, h_input); // 启动内核
    cudaDeviceSynchronize();

    unsigned int h_output;
    cudaMemcpy(&h_output, d_output, sizeof(unsigned int), cudaMemcpyDeviceToHost); // 拷贝结果到主机
    std::cout << "Highest bit position: " << h_output << std::endl; // 输出结果

    cudaFree(d_output);
    return 0;
}

// 输出是19
// 0000 0000 0000 1000 0000 0000 0000 0000
// 从低位的0开始数  起始为0