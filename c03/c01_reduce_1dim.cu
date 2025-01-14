//
// Created by Gary27 on 2025/1/6.
//

#include "cmdline.h"
#include <helper.h>
#include <iostream>
#include <vector>

__global__ void kernel_reduce_1dim(float *d_in, float *d_out, int n) {

    const auto tid = threadIdx.x;
    const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    float *d_in_block = d_in + blockDim.x * blockIdx.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {  // 限制线程的id为2的倍数
            d_in_block[tid] += d_in_block[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) d_out[blockIdx.x] = d_in_block[0];
}

__global__ void kernel_reduce_1dim_v2(float *d_in, float *d_out, int n) {
    const auto tid = threadIdx.x;
    const auto global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_tid >=n) return;
    float *d_in_block = d_in + blockIdx.x * blockDim.x;  // 必须有 用于区分block的行为  // 当前核函数相当于定义了一个block中各个线程的行为
    for (unsigned int i = 1, j = 2; i < blockDim.x; i *= 2, j *= 2) {
        // 公式： a = a + b
        // i: 当前thread需要相加的两个数的地址偏移量
        // j: threadIdx.x到a在数组中的索引的倍数
        auto ori = tid * j;  // 当前线程的offset从1 2 4 8 16 ...
        auto bias = ori + i;  // tid到数据的索引 从*2(0->0 1->2 2->4 ...) *4(0->0 1->4 2->8)
        if(bias < blockDim.x){  // bias 最终轮是当前block大小的一般 bias 一直要比ori大，所以首先check bias的边界
            d_in_block[ori] += d_in_block[bias];
        }
        __syncthreads();
    }
    //
    if (tid == 0) d_out[blockIdx.x] = d_in_block[0];
}

// version 3
// 交错配对的规约 就是0和64求和 1和65求和  初始步长为bs的一半然后逐步减小至1
__global__ void kernel_reduce_1dim_v3(float* din, float* dout, int n){
    const auto tid = threadIdx.x;
    const auto global_tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_tid >= n) return;
    float *din_block = din + blockDim.x * blockIdx.x;
    for (auto stride=blockDim.x / 2;stride >= 1; stride>>=1){
        auto bias = tid + stride;
        if(bias < blockDim.x){
            din_block[tid] += din_block[bias];
        }
        __syncthreads();
    }
    if (tid == 0) dout[blockIdx.x] = din_block[0];
}



int main() {
    cmdline::parser a;
    a.add<int>("version", 'v', "reduce algo version", false, 0);



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
    dim3 gs{(N + bsx - 1) / bsx, 1, 1};
    if (av == 1){
        kernel_reduce_1dim<<<gs, bs>>>(d_in, d_out, N);
    } else if(av == 2) {
        kernel_reduce_1dim_v2<<<gs, bs>>>(d_in, d_out, N);
    } else {
        kernel_reduce_1dim<<<gs, bs>>>(d_in, d_out, N);

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