//
// Created by Gary27 on 2025/3/21.
//
#include <cuda_runtime.h>
#include <cstdio>

__global__ void activemask_kernel() {
    unsigned int mask = __activemask();

    if (threadIdx.x == 0) {
        printf("Warp activemask: %08x\n", mask);
    }
}

__global__ void test_activemask() {
    // 强制线程 0 不活跃（条件始终为假）
    if (threadIdx.x != 0) {
        // 获取活跃线程掩码
        unsigned mask = __activemask();
        int tid = static_cast<int>(threadIdx.x);
        printf("CUDA: Thread %d, Active Mask = 0x%08X\n", tid, mask);
    }
}

__global__ void test_dynamic_mask() {
    int tid = threadIdx.x;
    // 仅允许偶數线程进入分支
    if (tid % 2 == 0) {
        unsigned mask = __activemask();  // CUDA
        // uint64_t mask = __activemask();  // ROCm
        printf("Thread %d: Active Mask = 0x%X\n", tid, mask);
    }
}

int main() {
    // activemask_kernel<<<1, 32>>>();
    // cudaDeviceSynchronize();

    test_activemask<<<1, 32>>>();  // 启动 32 线程的 Warp
    cudaDeviceSynchronize();

    test_dynamic_mask<<<1, 32>>>();  // 启动 32 线程的 Warp
    cudaDeviceSynchronize();


    return 0;
}