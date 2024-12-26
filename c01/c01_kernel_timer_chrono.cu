//
// Created by Gary27 on 2024/12/23.
//

#include <chrono>
#include <iostream>
#include <array>
#include <cuda_runtime.h>
#include "aux.h"
#include "helper.h"

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

int main(){

  const int N = 1024;
  constexpr int array_bytes_size = N * sizeof(float);
  std::array<float, N> ha, hb, hc;

  initializeData<float, N>(ha, 1.);
  initializeData<float, N>(hb, 1.);
  initializeData<float, N>(hc, 0.);

  float *da, *db, *dc;

  const int block_size = 256;
  const int grid_size = (N + block_size - 1) / block_size;

  auto t0 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMalloc(&da, array_bytes_size));
  CUDA_CHECK(cudaMalloc(&db, array_bytes_size));
  CUDA_CHECK(cudaMalloc(&dc, array_bytes_size));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), array_bytes_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), array_bytes_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, hc.data(), array_bytes_size, cudaMemcpyHostToDevice));
  auto t1 = std::chrono::high_resolution_clock::now();
  vectorAdd<<<grid_size, block_size>>>(da, db, dc, N);
  synchronizeDeviceIfNecessary();
  auto t2 = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(hc.data(), dc, array_bytes_size, cudaMemcpyDeviceToHost));
  auto t3 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));

  auto t_io_h2d = t1 - t0;
  auto t_kernel_exec = t2 - t1;
  auto t_io_d2h = t3 - t2;
  auto t_total = t3 - t0;
  std::cout << "io_h2d: " << std::chrono::duration_cast<std::chrono::microseconds>(t_io_h2d).count() << "us" << std::endl;
  std::cout << "kernel_exec: " << std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_exec).count() << "us" << std::endl;
  std::cout << "io_d2h: " << std::chrono::duration_cast<std::chrono::microseconds>(t_io_d2h).count() << "us" << std::endl;
  std::cout << "total: " << std::chrono::duration_cast<std::chrono::microseconds>(t_total).count() << "us" << std::endl;

  return 0;
}
/*
io_h2d: 181552us
kernel_exec: 103us
io_d2h: 16us
total: 181671us
 */