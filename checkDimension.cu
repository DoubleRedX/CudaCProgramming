#include <cstdio>
#include <cuda_runtime.h>

__global__ void checkIdx(void) {
    printf("threadIdx: (%d, %d, %d), blockIdx: (%d, %d, %d), blockDim: (%d, %d, %d), gridDim: (%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z
           );


//    printf("blockIdx: (%d, %d, %d) ", blockIdx.x, blockIdx.y, blockIdx.z);
//    printf("blockDim: (%d, %d, %d) ", blockDim.x, blockDim.y, blockDim.z);
//    printf("gridDim: (%d, %d, %d) ", gridDim.x, gridDim.y, gridDim.z);
}

int main(){
    int nElem = 6;
    dim3 block_size {3};  //
    dim3 grid_size {(nElem + block_size.x - 1) / block_size.x};  //

    printf("block_size: (%d, %d, %d)\n", block_size.x, block_size.y, block_size.z);
    printf("grid_size: (%d, %d, %d)\n", grid_size.x, grid_size.y, grid_size.z);

    checkIdx<<<block_size, grid_size>>>();
    cudaDeviceReset();
    return 0;
}