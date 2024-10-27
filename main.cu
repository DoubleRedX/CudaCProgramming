#include <cstdio>
#include <iostream>
#include <chrono>
#include "helper.h"


__global__ void ss(){
    if(threadIdx.x == 5){
        printf("hello cuda! thread %d", threadIdx.x);
    }
}

int main(){
    ss<<<1, 10>>>();

//    std::chrono::

//    cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}