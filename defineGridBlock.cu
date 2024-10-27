//
// Created by Gary27 on 2024/2/6.
//
#include <cuda_runtime.h>
#include <cstdio>


int main(){
    int nElem = 1024;

    dim3 block_size {1024};
    dim3 grid_size {(nElem + block_size.x - 1) / block_size.x};
    printf("grid.x %d block.x %d \n", grid_size.x, block_size.x);

    block_size.x = 512;
    grid_size = (nElem + block_size.x - 1) / block_size.x;
    printf("grid.x %d block.x %d \n", grid_size.x, block_size.x);

    block_size.x = 256;
    grid_size = (nElem + block_size.x - 1) / block_size.x;
    printf("grid.x %d block.x %d \n", grid_size.x, block_size.x);

    block_size.x = 128;
    grid_size = (nElem + block_size.x - 1) / block_size.x;
    printf("grid.x %d block.x %d \n", grid_size.x, block_size.x);

//    cudaDeviceReset();
    return 0;
}