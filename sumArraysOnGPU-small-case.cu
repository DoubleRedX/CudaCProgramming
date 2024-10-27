#include <cuda_runtime.h>
#include <cstdio>
#include <ctime>
#include <iostream>
//#include <format>
#include "helper.h"


void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int) time(&t));

    for(int i=0; i < size; ++i){
        ip[i] = (float )(rand() & 0xFFFF) / 10.0f;  // 8bit 一个字节0xFF
    }
}

void sumArrayOnHost(float *A, float *B, float *C, const int N){
    for(int idx=0;idx < N;++idx) C[idx] = A[idx] + B[idx];
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, int N){
    auto i = threadIdx.x;
    if(i < N) C[i] = A[i] + B[i];
}

int main(int argc, char **argv){
    printf("%s Starting ... \n", argv[0]);

    // setup device
    int dev = 0;
    cudaSetDevice(dev);

    // set up size of vectors
    int nElem = 32;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);
    auto h_A = (float *) malloc(nBytes);
    auto h_B = (float *) malloc(nBytes);
    auto host_C = (float *) malloc(nBytes);
    auto gpu_C = (float *) malloc(nBytes);

    // initial data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(host_C, 0, nBytes);
    memset(gpu_C, 0, nBytes);

    // malloc GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **) &d_A, nBytes);
    cudaMalloc((float **) &d_B, nBytes);
    cudaMalloc((float **) &d_C, nBytes);

    // transfer data
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // Configuration
    dim3 nThread {32};
    dim3 nBlock {(nElem + nThread.x - 1) / nThread.x};

    // calculate on Device
    sumArrayOnGPU<<<nBlock, nThread>>>(d_A, d_B, d_C, nElem);
    std::cout << "Execution configuration <<<" << nBlock.x << ", " << nThread.x <<  ">>>.\n";
//    std::cout << std::format("Execution configuration <<<{},{}>>>\n", nBlock.x, nThread.x) << std::endl;
    cudaMemcpy(gpu_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    // calculate on Host
    sumArrayOnHost(h_A, h_B, host_C, nElem);

    // check Result
    checkResult(host_C, gpu_C, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(host_C);
    free(gpu_C);

    return 0;
}