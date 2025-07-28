//
// Created by Gary27 on 2025/3/24.
//
#include <cstdio>
#include <iostream>

__global__ void shfl_xor_sync_example() {
    // 每个线程的初始值为其线程索引
    int value = threadIdx.x;

    // 打印执行 __shfl_xor_sync 之前的值
    printf("Before __shfl_xor_sync: Thread %d has value %d\n", threadIdx.x, value);

    // 使用 __shfl_xor_sync 进行数据交换
    // mask: 0xFFFFFFFF 表示所有线程都参与 shuffle 操作
    // delta: 1 表示从当前线程向前移动 1 个线程
    unsigned mask = 0xFFFFFFFF;
    int delta = 2;
    int shuffled_value = __shfl_xor_sync(mask, value, delta);
    // 打印执行 __shfl_xor_sync 之后的值
    printf("After __shfl_xor_sync: Thread %d has value %d\n", threadIdx.x, shuffled_value);
}

int main() {
    // 启动一个包含单个 warp 的 kernel (32 threads)
    std::cout << "============================block size 32 case===========================" << "\n";
    shfl_xor_sync_example<<<1, 32>>>();
    cudaDeviceSynchronize();
    std::cout << "============================block size 64 case===========================" << "\n";
    shfl_xor_sync_example<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}