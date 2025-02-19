//
// Created by Gary27 on 2025/2/11.
//

// memTransfer

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

int main(){
    int dev = 1;
    cudaSetDevice(dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    unsigned int data_size = 1<<28;
    auto n_bytes = data_size * sizeof(float );

    std::cout << "Device " << dev << ": " << prop.name << " memory size " << data_size << " nbyte " << (n_bytes/1024.0f/1024.0f) << "MB\n";

    // allocate host mem
    auto *hst_mem = new float[data_size];
    // allocate device mem
    float *da;
    cudaMalloc(&da, n_bytes);
    // initialize host mem
    std::fill_n(hst_mem, data_size, 0.5f);  // or for(int i=0;i<int_size;++i) h_mem[i] = 0.5f;


    // allocate device mem
    float* dev_mem;
    cudaMalloc(&dev_mem, n_bytes);

    // transfer data from host to device
    cudaMemcpy(dev_mem, hst_mem, data_size, cudaMemcpyHostToDevice);

    // transfer data from device to host
    cudaMemcpy(hst_mem, dev_mem, data_size, cudaMemcpyDeviceToHost);

    // free mem
    delete[] hst_mem;
    cudaFree(dev_mem);
    cudaDeviceReset();
    return 0;
}