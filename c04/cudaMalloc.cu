//
// Created by Gary27 on 2025/2/11.
//

// memTransfer

#include <cuda_runtime.h>


int main(){
    unsigned int int_size = 1<<22;
    unsigned int nbytes = int_size * sizeof(float );
    auto *ha = (float*)malloc(nbytes);
    float *da;
    cudaMalloc((float**)&da, nbytes);

    free(ha);
    return 0;
}