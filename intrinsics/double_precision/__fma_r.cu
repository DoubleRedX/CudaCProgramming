//
// Created by Gary27 on 2025/1/17.
//
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

__global__ void fma_example(double* result) {
    double a = 1.523224234652346235462242443;
    double b = 2.52224242422562356256443;
    double c = 3.54242422562562562564423;

    result[0] = __fma_rn(a, b, c); // 舍入到最接近的偶数
    result[1] = __fma_rd(a, b, c); // 向负无穷方向舍入
    result[2] = __fma_ru(a, b, c); // 向正无穷方向舍入
    result[3] = __fma_rz(a, b, c); // 向零方向舍入
}

int main() {
    const int num_results = 4;
    double h_result[num_results];
    double* d_result;

    // 分配设备内存
    cudaMalloc(&d_result, num_results * sizeof(double));

    // 启动内核
    fma_example<<<1, 1>>>(d_result);

    // 将结果拷贝回主机
    cudaMemcpy(h_result, d_result, num_results * sizeof(double), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << std::fixed << std::setprecision(50);
    std::cout << "__fma_rn: " << h_result[0] << std::endl; // 舍入到最接近的偶数
    std::cout << "__fma_rd: " << h_result[1] << std::endl; // 向负无穷方向舍入
    std::cout << "__fma_ru: " << h_result[2] << std::endl; // 向正无穷方向舍入
    std::cout << "__fma_rz: " << h_result[3] << std::endl; // 向零方向舍入

    // 释放设备内存
    cudaFree(d_result);

    return 0;
}