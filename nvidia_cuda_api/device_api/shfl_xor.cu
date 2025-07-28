//
// Created by Gary27 on 2025/3/24.
//
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>


__global__ void test_shfl_xor_sync_32(int* output32) {
    int lane_id = threadIdx.x % 32;
    int value = lane_id;

    // 测试分组宽度为32，掩码为全活跃（0xFFFFFFFFFFFFFFFF）
//    int result32 = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, var, 1, 32);
    int result32 = __shfl_xor_sync(0xFFFFFFFFU, value, 1, 32);  // 0x000x7ffefcbfd050
    output32[lane_id] = result32;
}

__global__ void test_shfl_xor_sync_64(int* output64) {
    int lane_id = threadIdx.x % 32;
    int value = lane_id;
    // 测试分组宽度为64，掩码为全活跃（0xFFFFFFFFFFFFFFFF）
    int result64 = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, var, 1, 64);
    // int result64 = __shfl_xor_sync(0xFFFFFFFFULL, value, 1, 64);  // 0x000x7ffefcbfd050
    output64[lane_id] = result64;
}

int main(){
    const int num_threads = 32;
    int *d_output32, *d_output64;
    int h_output32[num_threads], h_output64[num_threads];

    // 分配设备内存
    cudaMalloc(&d_output32, num_threads * sizeof(int));
    cudaMalloc(&d_output64, num_threads * sizeof(int));

    // 启动内核
//    hipLaunchKernelGGL(test_shfl_xor_sync, 1, num_threads, 0, 0, d_output32, d_output64);
    test_shfl_xor_sync_32<<<1, num_threads>>>(d_output32);

    // 复制回主机
    cudaMemcpy(h_output32, d_output32, num_threads * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output64, d_output64, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证分组宽度为32的行为（与CUDA的warpSize=32对比）
    bool valid32 = true;
    for (int i = 0; i < 32; ++i) {
        int expected = (i ^ 1) % 32; // 分组内异或，超出32的线程索引回绕
        if (h_output32[i] != expected) {
            std::cout << "i: " << i << " with host value: " << expected << ", device value: " << h_output32[i] << '\n';
            valid32 = false;
            break;
        }
    }
    std::cout << "0x" << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << h_output32 << std::endl;
    std::cout << "Width=32: " << (valid32 ? "PASS" : "FAIL") << std::endl;

    // 验证分组宽度为64的行为（与ROCm的warpSize=64对比）
    bool valid64 = true;
    for (int i = 0; i < num_threads; ++i) {
        int expected = i ^ 1; // 全部64线程参与异或
        if (h_output64[i] != expected) {
            std::cout << "i: " << i << " with host value: " << expected << ", device value: " << h_output32[i] << '\n';
            valid64 = false;
            break;
        }
    }
    std::cout << "0x" << std::hex << std::uppercase << std::setw(16) << std::setfill('0') << h_output64 << std::endl;
    std::cout << "Width=64: " << (valid64 ? "PASS" : "FAIL") << std::endl;

    // 清理
    cudaFree(d_output32);
    cudaFree(d_output64);

    unsigned mask   = 0xFFFFFFFFU;
    std::cout << "mask 32: " << mask << "\n";
    auto mask_ull = (unsigned long long)mask;
    std::cout << "mask 64: " << mask_ull << "\n";

    return 0;
}