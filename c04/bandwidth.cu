//
// Created by Gary27 on 2025/3/7.
//

#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int device_id = 0;  // 选择设备ID（默认为0）
    cudaDeviceProp prop {};
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);

    if (err != cudaSuccess) {
        fprintf(stderr, "无法获取设备属性: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 计算理论内存带宽 (GB/s)
    // 公式：带宽 = (总线宽度(bit) / 8) * 内存时钟频率(kHz -> Hz) * 2(DDR) / 1e9
    float bandwidth_GBps = (prop.memoryBusWidth / 8.0f) *
                           prop.memoryClockRate * 1e3f * 2 /
                           1e9f;

    printf("设备 %d: %s\n", device_id, prop.name);
    printf("内存总线宽度: %d 位\n", prop.memoryBusWidth);
    printf("内存时钟频率: %d MHz\n", prop.memoryClockRate / 1000);
    printf("理论内存带宽: %.2f GB/s\n", bandwidth_GBps);

    return 0;
}