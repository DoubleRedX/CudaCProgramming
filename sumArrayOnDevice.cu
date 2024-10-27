#include <cstdlib>
#include <string>
#include <ctime>
#include <iostream>
//#include <cuda_runtime.h>

void initialData(float *ip, int size){
    time_t t;
    srand((unsigned int)time(&t));
    for(int i=0;i<size;++i){
        ip[i] = (float)( rand() & 0xFF ) / 10.0F;
    }
}

__global__ void sumArrayOnDevice(float *A, float *B, float *C, const int N){

    C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];

}

int main(){

    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    auto h_A = (float *) malloc(nBytes);
    auto h_B = (float *) malloc(nBytes);
    auto h_C = (float *) malloc(nBytes);
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArrayOnDevice<<<1, nElem>>>(d_A, d_B, d_C, nElem);

    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);

    for(int i=0;i<nElem;++i){
        std::cout << "i: " << h_C[i] << "\n";
    }

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}