//
// Created by Gary27 on 2025/1/21.
//

#include <cstdio>

__global__ void shuffle_example(int *output) {
    int lane_id = threadIdx.x % 32;  // 获取线程在当前 warp 中的 lane ID
    int value = lane_id;             // 每个线程的初始值为其 lane ID

    // 使用 __shfl_sync 将 lane_id 为 0 的线程的值广播给 warp 中的所有线程
    int shuffled_value = __shfl_sync(0xFFFFFFFF, value, 0);

    // 将结果存储到全局内存中
    output[threadIdx.x] = shuffled_value;
}

int main() {
    const int num_threads = 32;
    int h_output[num_threads];
    int *d_output;

    // 分配设备内存
    cudaMalloc((void**)&d_output, num_threads * sizeof(int));

    // 启动内核
    shuffle_example<<<1, num_threads>>>(d_output);

    // 将结果拷贝回主机
    cudaMemcpy(h_output, d_output, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Shuffled values:\n");
    for (int i = 0; i < num_threads; ++i) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // 释放设备内存
    cudaFree(d_output);

    return 0;
}