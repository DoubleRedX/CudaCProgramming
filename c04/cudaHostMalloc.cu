//
// Created by Gary27 on 2025/2/19.
//

#include <iostream>
#include <cuda_runtime.h>

int main(){
    int dev = 1;
    cudaSetDevice(dev);

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    unsigned int data_size = 1 << 28;
    auto n_bytes = data_size * sizeof(float);

    std::cout << "Device: " << prop.name << " mem size " << data_size << " " << n_bytes/1024/1024 << "MB\n";

    // malloc host pin mem & device mem
    float *hst_mem;
    cudaMallocHost(&hst_mem, n_bytes);

    // initialize host mem
    std::fill_n(hst_mem, data_size, 0.5f);  // or for(int i=0;i<int_size;++i) h_mem[i] = 0.5f;

    float *dev_mem;
    cudaMalloc(&dev_mem, n_bytes);

    // transfer mem
    cudaMemcpy(dev_mem, hst_mem, n_bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(hst_mem, dev_mem, n_bytes, cudaMemcpyDeviceToHost);

    cudaFreeHost(hst_mem);
    cudaFree(dev_mem);
    cudaDeviceReset();
    return 0;
}
// 129382924
// 87820895

// 跑一个实验看看接口的影响？

// 目前看来拷贝的内存足够大的时候。cudaMallocHost才起作用.(1024MB)否则是不如cudaMalloc的