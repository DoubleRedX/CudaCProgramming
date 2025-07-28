//
// Created by Gary27 on 2025/3/21.
//
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cstdio>

__global__ void shfl_up_sync_example() {
    int lane_id = threadIdx.x % 32;
    int value = lane_id;

    // 条件：仅偶数线程活跃
    bool is_active = (lane_id % 2 == 0);

    // 获取活跃线程的掩码
    unsigned int mask = __activemask();

    // 使用 __shfl_up_sync（显式指定掩码）
    int shuffled_value = __shfl_up_sync(mask, value, 1);

    // 打印结果（仅活跃线程打印）
    if (is_active) {
        printf("Thread %2d (__shfl_up_sync): received %2d\n", lane_id, shuffled_value);
    }
}

int main() {
    shfl_up_sync_example<<<1, 32>>>(); // 启动 1 个线程块，32 个线程
    cudaDeviceSynchronize();
    return 0;
}