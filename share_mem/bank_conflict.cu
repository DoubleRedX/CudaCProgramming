//
// Created by Gary27 on 2025/7/3.
//

#include <cuda_runtime.h>
#include <cstdio>

__global__ void bank_conflict_kernel(int* output) {
    __shared__ int sh_array[32];
    unsigned int tid_x = threadIdx.x;
    sh_array[tid_x % 4] = tid_x;
    __syncthreads();
    __threadfence();
    __threadfence_block();
    __threadfence_system();

    output[tid_x] = sh_array[tid_x % 4];
}


int main() {

    int h_array[32];
    int* d_output;
    cudaMalloc(&d_output, sizeof(int) * 32);

    bank_conflict_kernel<<<1, 32>>>(d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_array, d_output, sizeof(int) * 32, cudaMemcpyDeviceToHost);

    for (int i=0;i<32;i++) {
        printf("Thread %d -> %d\n ", i, h_array[i]);
    }
    cudaFree(d_output);
    return 0;
}

// profile
// ncu

// ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum, l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__warps_launched.sum ./bank_conflict

/*
 * 分析发现没有发生bank冲突。原因是，新的架构能够自动检测bank conflict，进行bank conflict mitigation(?)，应该是被优化掉了。但是具体的情况不了解，可以再进一步查阅资料了解。
 *
 *
 */