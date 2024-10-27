//
// Created by Gary27 on 2024/2/7.
//

#ifndef CUDACPROGRAMMING_HELPER_H
#define CUDACPROGRAMMING_HELPER_H

#include <cmath>

#define CHECK(call) { \
    const cudaError_t error = call; \
    if(error != cudaSuccess){       \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);      \
    } \
}

void checkResult(float *hostRef, float  *gpuRef, const int N){
    double epsilon = 1.0E-8;
    int match = 1;
    for(int i=0;i<N;++i){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if(match == 1) printf("Arrays match.\n\n");
}





#endif //CUDACPROGRAMMING_HELPER_H
