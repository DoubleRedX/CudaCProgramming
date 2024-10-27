#include <iostream>
#include <cuda_runtime.h>
#include "helper.h"


template<typename T>
void initialData(T* data, int size){
    for(int i=0;i<size;++i) data[i] = (float)i;
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    auto ia = A, ib = B, ic = C;
    for(int i=0;i<ny;++i){
        for(int j=0;j<nx;++j){
            ic[j] = ia[j] + ib[j];
        }
        ia += nx;
        ib += nx;
        ic += ny;
    }
}

// 二维网格 二维块
__global__ void sumMatrixOnDevice(const float *A, const float *B, float *C, const int nx, const int ny){
    auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    auto idx = ix + iy * nx;
    if(ix < nx && iy < ny) C[idx] = A[idx] + B[idx];  // 越界检测
}

// 还没计算
__global__ void sumMatrixOnDevice1DGrid1DBlock(const float *A, const float *B, float *C, const int nx, const int ny){
    auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    if(ix < nx){
        for(int j=0;j<ny;++j){
            auto idx = j * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// 还没计算
__global__ void sumMatrixOnDevice2DGrid1DBlock(const float *A, const float *B, float *C, const int nx, const int ny){
    auto ix = threadIdx.x + blockDim.x * blockIdx.x;
    auto iy = blockIdx.y;
    auto idx = iy * nx + ix;
    if(ix < nx && iy < ny) C[idx] = A[idx] + B[idx];
}


int main(int argc, char **argv){
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp {};
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size
    int nx = 1 << 14;
    int ny = 1 << 14;  // 24太大了
    int nxy = nx * ny;
    auto nBytes = nxy * sizeof(float);

    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    auto h_A = (float*) malloc(nBytes);
    auto h_B = (float*) malloc(nBytes);
    auto hostRes = (float*) malloc(nBytes);
    auto gpuRes = (float*) malloc(nBytes);

    // initial data
    initialData<float>(h_A, nxy);
    initialData<float>(h_B, nxy);
    memset(hostRes, 0, nBytes);
    memset(gpuRes, 0, nBytes);

    sumMatrixOnHost(h_A, h_B, hostRes, nx, ny);

    // malloc device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, nBytes);
    cudaMalloc((void**)&d_B, nBytes);
    cudaMalloc((void**)&d_C, nBytes);

    // transfer data to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // execution configuration
    dim3 block_size {32, 32};
    dim3 grid_size {(nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y};
    sumMatrixOnDevice<<<grid_size, block_size>>>(d_A, d_B, d_C, nx, ny);
    cudaDeviceSynchronize();

    cudaMemcpy(gpuRes, d_C, nBytes, cudaMemcpyDeviceToHost);
    checkResult(hostRes, gpuRes, nxy);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRes);
    free(gpuRes);

    cudaDeviceReset();
    return 0;
}