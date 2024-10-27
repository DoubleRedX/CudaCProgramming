#include "cuda_runtime.h"
#include <cstdio>

#define CHECK(call){ \
    const cudaError_t error = call; \
    if(error != cudaSuccess){ \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10 * error); \
    }}

void initialInt(int *arr, int size){
    for(int i=0;i<size;++i) arr[i] = i;
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d * %d) \n", nx, ny);
    for(int iy=0;iy<ny;++iy){
        for(int ix=0;ix<nx;++ix){
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    auto idx = nx * iy + ix;
    printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}



int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // get device info
    int dev = 0;
    cudaDeviceProp deviceProp {};
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set Matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    auto nBytes = nxy * sizeof(float);

    // malloc host memory
    auto *h_A = (int *) malloc(nBytes);
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int *d_A {nullptr};
    cudaMalloc((int**)&d_A, nBytes);
    // transfer data
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    // set up execution configuration
    dim3 block_size {4, 2};
    dim3 grid_size {(nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y};
    // invoke kernel function
    printThreadIndex<<<grid_size, block_size>>>(d_A, nx, ny);
    cudaDeviceSynchronize();

    // free memory
    cudaFree(d_A);
    free(h_A);

    // reset device
    cudaDeviceReset();
    return 0;
}