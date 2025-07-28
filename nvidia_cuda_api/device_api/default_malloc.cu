//
// Created by Gary27 on 2025/4/9.
//
//
// Created by Gary27 on 2025/4/9.
//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int main()
{
    const size_t bufferSize = 1024 * 1024; // 1MB测试缓冲区
    // const int patternToCheck = 0x01010101; // 需要验证的魔数
    const int patternToCheck = 0x00000000; // 也可以尝试验证零值的情况

    // 1. 分配未初始化的GPU内存
    int *d_buffer = nullptr;
    cudaError_t err = cudaMalloc(&d_buffer, bufferSize);
    if (err != cudaSuccess || d_buffer == nullptr) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 2. 将未初始化的GPU内存拷贝到主机
    std::vector<int> h_buffer(bufferSize / sizeof(int));
    err = cudaMemcpy(h_buffer.data(), d_buffer, bufferSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_buffer);
        return 1;
    }

    // 3. 分析内存内容
    size_t matchCount = 0;
    size_t mismatchCount = 0;
    const int maxSamplesToShow = 10;

    for (size_t i = 0; i < h_buffer.size(); ++i) {
        if (h_buffer[i] == patternToCheck) {
            ++matchCount;
        } else {
            ++mismatchCount;
            if (mismatchCount <= maxSamplesToShow) {
                std::cout << "Mismatch at index " << i
                          << ": 0x" << std::hex << h_buffer[i]
                          << " (" << std::dec << h_buffer[i] << ")\n";
            }
        }
    }


    // 4. 打印统计结果
    std::cout << "\n=== Memory Analysis Report ===" << std::endl;
    std::cout << "Total elements:     " << h_buffer.size() << std::endl;
    std::cout << "Matching 0x" << std::hex << patternToCheck << std::dec
              << ":    " << matchCount << " ("
              << (matchCount * 100.0 / h_buffer.size()) << "%)\n";
    std::cout << "Mismatches:        " << mismatchCount << " ("
              << (mismatchCount * 100.0 / h_buffer.size()) << "%)\n";

    // 5. 清理资源
    cudaFree(d_buffer);
    return 0;
}