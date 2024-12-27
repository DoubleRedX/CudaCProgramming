//
// Created by Gary27 on 2024/2/7.
//

#ifndef CUDACPROGRAMMING_HELPER_H
#define CUDACPROGRAMMING_HELPER_H

#include <cmath>
#include <iostream>

#define CHECK(call) { \
    const cudaError_t error = call; \
    if(error != cudaSuccess){       \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1);      \
    } \
}

constexpr int CUDA_COMPUTE_CAPABILITY_900 = 900;

inline void synchronizeDeviceIfNecessary() {
#if __CUDA__ARCH__ >= CUDA_COMPUTE_CAPABILITY_900
    cudaDeviceSynchronize();
#else
    printf("Running on sm_90+ or newer. Synchronization skipped.\n");
#endif

}

inline void checkResult(float *hostRef, float  *gpuRef, const int N){
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


#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


#endif //CUDACPROGRAMMING_HELPER_H
