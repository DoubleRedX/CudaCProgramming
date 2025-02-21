//
// Created by Gary27 on 2025/2/21.
//
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

int main(){

    int dev_idx = 1;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev_idx);

    spdlog::info("Device Name: {}", prop.name);
    spdlog::info("L2 Cache Size: {} MiB", prop.l2CacheSize / 1024 / 1024);


    return 0;
}