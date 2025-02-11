//
// Created by Gary27 on 2025/1/16.
//
#include <iostream>
#include <cuda_runtime.h> // CUDA 运行时库

// CUDA Kernel 函数：使用 PTX 汇编指令 mul.wide.u32
__global__ void wideMultiplyKernel(const uint32_t *A, const uint32_t *B, uint64_t *C, int N) {
    // 计算当前线程的全局索引
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保索引在有效范围内
    if (idx < N) {
        uint32_t a = A[idx]; // 第一个 32 位操作数
        uint32_t b = B[idx]; // 第二个 32 位操作数
        uint64_t c;          // 64 位结果

        // 使用 PTX 汇编指令 mul.wide.u32
        asm("mul.wide.u32 %0, %1, %2;" : "=l"(c) : "r"(a), "r"(b));

        // 将结果存储到输出数组
        C[idx] = c;
    }
}

__device__ void print_binary(uint64_t num) {
    // 打印 64 位数的二进制表示
    for (int i = 63; i >= 0; --i) {
        printf("%lu", (num >> i) & 1);
        if (i % 8 == 0) {
            printf(" ");  // 每8位添加一个空格（便于查看）
        }
    }
}

__global__ void mul_u32_kernel(const uint32_t A, const uint32_t B, uint64_t *C) {
    printf("A in binary: ");
    print_binary(A);
    printf("\n");
    printf("B in binary: ");
    print_binary(B);
    printf("\n");
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(*C) : "r"(A), "r"(B));
    printf("C in binary: ");
    print_binary(*C);
    printf("\n");
    printf("C: %ld", *C);

}

int main() {
    if (true) {
        uint32_t a = 0x22222222;
        uint32_t b = 0x22222222;
        uint64_t *C;
        uint64_t host_result = (uint64_t)a * b;  // 主机端计算结果
        cudaMalloc(&C, sizeof(uint64_t));
        mul_u32_kernel<<<1, 1>>>(a, b, C);
        uint64_t res;
        cudaMemcpy(&res, C, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        if(host_result == res){
            std::cout << "right ans" << std::endl;
            std::cout << "device res: " << res << std::endl;
            std::cout << "host  res: " << host_result << std::endl;
        }else {
            std::cout << "wrong ans" << std::endl;
            std::cout << "device res: " << res << std::endl;
            std::cout << "host  res: " << host_result << std::endl;
        }
        auto res32 = static_cast<uint32_t>(res);
        std::cout << "res32: " << res32 << std::endl;
    } else {

        // 定义向量大小
        int N = 1024; // 假设向量长度为 1024
        size_t size32 = N * sizeof(uint32_t);
        size_t size64 = N * sizeof(uint64_t);

        // 分配主机内存
        auto *h_A = (uint32_t *) malloc(size32);
        auto *h_B = (uint32_t *) malloc(size32);
        auto *h_C = (uint64_t *) malloc(size64);

        // 初始化主机数据
        for (int i = 0; i < N; i++) {
            h_A[i] = static_cast<uint32_t>(i);      // A = [0, 1, 2, ..., 1023]
            h_B[i] = static_cast<uint32_t>(i * 2);  // B = [0, 2, 4, ..., 2046]
        }

        // 分配设备内存
        uint32_t *d_A, *d_B;
        uint64_t *d_C;
        cudaMalloc(&d_A, size32);
        cudaMalloc(&d_B, size32);
        cudaMalloc(&d_C, size64);

        // 将数据从主机复制到设备
        cudaMemcpy(d_A, h_A, size32, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size32, cudaMemcpyHostToDevice);

        // 定义线程块大小和网格大小
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // 调用 CUDA Kernel
        wideMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // 将结果从设备复制回主机
        cudaMemcpy(h_C, d_C, size64, cudaMemcpyDeviceToHost);

        // 验证结果
        bool success = true;
        for (int i = 0; i < N; i++) {
            uint64_t expected = static_cast<uint64_t>(h_A[i]) * static_cast<uint64_t>(h_B[i]);
            if (h_C[i] != expected) {
                success = false;
                std::cerr << "Error at index " << i << ": expected " << expected << ", got " << h_C[i] << std::endl;
                break;
            }
        }

        if (success) {
            std::cout << "Kernel executed successfully!" << std::endl;
        }

        // 释放设备内存
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // 释放主机内存
        free(h_A);
        free(h_B);
        free(h_C);

    }

    return 0;
}