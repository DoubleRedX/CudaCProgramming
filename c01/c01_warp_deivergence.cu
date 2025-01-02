//
// Created by Gary27 on 2025/1/2.
//

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <helper.h>

// p69

#define WARP_SIZE 32
#define N 1024

// warmup kernel
// 消除启动的引导阶段开销

__global__ void warmup_kernel() {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    auto x = tid * 10;
  }
}

// kernel with warp divergence
__global__ void divergent_kernel(int *output) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    if (tid % 2 == 0) {
      output[tid] = 2;
    } else {
      output[tid] = 3;
    }
  }
}

// kernel w/o warp divergence (根据warpSize对齐分支)
__global__ void non_divergent_kernel(int *output) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto warp_id = tid / WARP_SIZE;  // 计算当前线程所属的线程束编号
  if (tid < N) {
    if (warp_id % 2 == 0) {  // 分支对齐到线程束级别
      output[tid] = 2;
    } else {
      output[tid] = 3;
    }
  }
}

int main(){
  std::vector<int> output1(N);
  std::vector<int> output2(N);
  int *doutput1, *doutput2;
  CUDA_CHECK(cudaMalloc(&doutput1, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&doutput2, N * sizeof(int)));
  dim3 bs {256, 1, 1};
  dim3 gs {(N + bs.x - 1) / bs.x, 1, 1};
  warmup_kernel<<<gs, bs>>>();
  synchronizeDeviceIfNecessary();
  divergent_kernel<<<gs, bs>>>(doutput1);
  synchronizeDeviceIfNecessary();
  non_divergent_kernel<<<gs, bs>>>(doutput2);
  CUDA_CHECK(cudaMemcpy(output1.data(), doutput1, N * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output2.data(), doutput2, N * sizeof(int), cudaMemcpyDeviceToHost));
  std::cout << "output1: " << output1[1] << std::endl;
  std::cout << "output2: " << output2[1] << std::endl;
  CUDA_CHECK(cudaFree(doutput1));
  CUDA_CHECK(cudaFree(doutput2));
  return 0;
}

// 1. nvprof --metrics branch_efficiency ./warp_divergence  //  ! 7.5 or higher nvprof is not supported.

// 2. nsys profile --stats=true ./warp_divergence

// 拉取的pytorch镜像（dockerpull.com/pytorch/pytorch 2.2.2-cuda11.8-cudnn8-devel）中没有nsys这个东西。但是11的机器上有。
// 镜像里有ncu这个工具，但是没用成功/

// 3.ncu --metrics branch_efficiency ./warp_divergence