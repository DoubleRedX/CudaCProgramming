//
// Created by Gary27 on 2025/7/1.
//

#include <cuda_runtime.h>
#include <cstdio>

__shared__ float global_shmem[24];      // 声明为全局的？

__global__ void myKernel() {
    /*
     *     // 动态共享内存, 在launch kernel的时候指定大小: myKernel<<<gridSize, blockSize, shmemBytes>>>();
     *     只能声明为1维的
     */
     extern __shared__ float dynamicShared[];
    /*
     *   // 静态共享内存， 编译期决定 1024 * 4 = 4096 bytes
     */
    __shared__ float staticShared[1024];
    dynamicShared[threadIdx.x] = staticShared[threadIdx.x];
    global_shmem[23] = staticShared[threadIdx.x] + 1;
    printf("global_shmem[23] = %f\n", global_shmem[23]);

}

/*
 * bank:
 * 一个bank大小为32位, 4个bytes。共计32个bank
 * 32位模式： bank_index = bytes_addr / 4 % 32
 * 64位模式： bank_index = bytes_addr / 8 % 32
 */

int main() {
    int deviceID = 0;
    int maxSharedMemPerBlock = 0;
    cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, deviceID);
    printf("[设备属性] Max shared memory per block: %d bytes\n", maxSharedMemPerBlock);    // 48KB 硬件物理上限

    int maxSharedMemPerBlockOptin = 0;
    cudaDeviceGetAttribute(&maxSharedMemPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID);
    printf("[设备属性] Max opt-in shared memory per block: %d bytes\n", maxSharedMemPerBlockOptin);    // 99KB 动态共享内存上限


    cudaFuncAttributes attr{};
    cudaFuncGetAttributes(&attr, myKernel);    // 查询kernel使用了多少shmem，决定kernel的launch config
    printf("[Kernel 编译属性] Static shared memory: %lu bytes\n", attr.sharedSizeBytes);    // ⚠️ 注意：attr.sharedSizeBytes 显示的是 静态共享内存用量，不包含你 launch 时传进去的动态大小。

    // 启动 kernel，并设置动态共享内存大小（例：512 floats = 2048 bytes）
    int dynamicSharedMemSize = 512 * sizeof(float);
    int threadsPerBlock = 1024;
    myKernel<<<1, threadsPerBlock, dynamicSharedMemSize>>>();
    // 等待执行结束
    cudaDeviceSynchronize();

    printf("[运行时] Kernel launched with dynamic shared memory size: %d bytes\n", dynamicSharedMemSize);


    return 0;
}