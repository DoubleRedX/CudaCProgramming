//
// Created by Gary27 on 2025/1/6.
//

#include <helper.h>
#include <iostream>
#include <vector>

__global__ void kernel_reduce_1dim(float *d_in, float *d_out, int n) {

  const auto tid = threadIdx.x;
  const auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) return;
  float *d_in_block = d_in + blockDim.x * blockIdx.x;
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      d_in_block[tid] += d_in_block[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) d_out[blockIdx.x] = d_in_block[0];
}

__global__

int main(){
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
  kernel_reduce_1dim<<<gs, bs>>>(d_in, d_out, N);
  CUDA_CHECK(cudaMemcpy(res.data(), d_out, bsx * sizeof(float), cudaMemcpyDeviceToHost));
  for (auto i : res) {
    sum += i;
  }
  std::cout << "sum: " << sum << std::endl;
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
}