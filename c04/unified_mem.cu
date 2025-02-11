//
// Created by Gary27 on 2025/2/11.
//
#include <cuda_runtime.h>

#include "spdlog/spdlog.h"

int main(){


    float* value;
    cudaMallocManaged(&value, sizeof(float));
    spdlog::info("value is {}", *value);
    *value = 100.0;
    spdlog::info("value is {}", *value);
    *value = 120.0;
    spdlog::info("value is {}", *value);
    *value += 100.0;
    spdlog::info("value is {}", *value);

    cudaFree(value);

    return 0;
}