//
// Created by Gary27 on 2025/1/15.
//
#include "cmdline.h"
#include <helper.h>
#include <iostream>
#include <vector>


__global__ void kernel_reduce_1dim_unroll(float* din, float* dout, int n){
    // 1 thread block(t1) deal 2 data block(b1 b2)
    auto tid = threadIdx.x;
    auto idx = blockIdx.x * blockDim.x * 2 + tid;  // thread block(t1) index to data block(d1)

    // data block start address
    auto din_block = din + blockIdx.x * blockDim.x * 2;

    // b1 += b2
    if(idx + blockDim.x < n) din[idx] += din[idx + blockDim.x];  // 这句怎么写应该都行//用tid算应该也一样  //这句不做boundary check吗？
    __syncthreads();  // 确保b1+b2完成

    for(auto stride=blockDim.x / 2;stride>=1;stride>>=1){
        if(tid + stride < blockDim.x){
            din_block[tid] += din_block[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) dout[blockIdx.x] = din_block[0];
}


__global__ void kernel_reduce_1dim_unroll_factor8(float* din, float* dout, int n){

    // 展开线程的规约
    auto tid = threadIdx.x;
    auto idx = tid + blockDim.x * blockIdx.x * 8;

    auto din_block = din + blockDim.x * blockIdx.x * 8;

    if(idx + 8 * blockDim.x < n){
        auto a0 = din[idx];
        auto a1 = din[idx + blockDim.x * 1];
        auto a2 = din[idx + blockDim.x * 2];
        auto a3 = din[idx + blockDim.x * 3];
        auto a4 = din[idx + blockDim.x * 4];
        auto a5 = din[idx + blockDim.x * 5];
        auto a6 = din[idx + blockDim.x * 6];
        auto a7 = din[idx + blockDim.x * 7];
        din[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    for(auto stride = blockDim.x / 2; stride > 32; stride >>=1){
        if(tid + stride < blockDim.x){
            din_block[tid] += din_block[tid + stride];
        }
        __syncthreads();
    }
    if(tid < 32){
        volatile auto *vmem = din_block;  // volatile强制写回内存
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];  // 循环展开
    }
    if(tid == 0) dout[blockIdx.x] = din_block[0];
}

int main() {
    cmdline::parser a;
    a.add<int>("factor", 'f', "unroll factor", false, 8);



    auto av = a.get<int>("version");
    constexpr int N = 1024;
    constexpr int bsx = 128;
    std::vector<float> data(N, 10.0);
    std::vector<float> res(bsx, 10.0);
    float sum = 0.0;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    dim3 bs{bsx, 1, 1};
    CUDA_CHECK(cudaMalloc(&d_out, bsx * sizeof(float)));
    if (av == 2){
        dim3 gs{((N + bsx - 1) / bsx) / 2, 1, 1};  // 此处将grid的大小缩小为原来的1/2  // 因为同样大小的数据只需要一半thread
        kernel_reduce_1dim_unroll<<<gs, bs>>>(d_in, d_out, N);
    } else if(av == 8) {
        dim3 gs{((N + bsx - 1) / bsx) / 8, 1, 1};  // 此处将grid的大小缩小为原来的1/8  // 因为同样大小的数据只需要1/8thread
        kernel_reduce_1dim_unroll_factor8<<<gs, bs>>>(d_in, d_out, N);
    } else {
        // 默认为2
        dim3 gs{((N + bsx - 1) / bsx) / 2, 1, 1};
        kernel_reduce_1dim_unroll<<<gs, bs>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaMemcpy(res.data(), d_out, bsx * sizeof(float), cudaMemcpyDeviceToHost));
    for (auto i: res) {
        sum += i;
    }
    std::cout << "sum: " << sum << std::endl;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}