//
// Created by Gary27 on 2024/12/26.
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
  cudaEvent_t t0, t1, t2, t3;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventCreate(&t2));
  CUDA_CHECK(cudaEventCreate(&t3));


  initializeData<float, N>(ha, 1.);
  initializeData<float, N>(hb, 1.);
  initializeData<float, N>(hc, 0.);

  float *da, *db, *dc;

  const int block_size = 256;
  const int grid_size = (N + block_size - 1) / block_size;

  CUDA_CHECK(cudaEventRecord(t0));
  CUDA_CHECK(cudaMalloc(&da, array_bytes_size));
  CUDA_CHECK(cudaMalloc(&db, array_bytes_size));
  CUDA_CHECK(cudaMalloc(&dc, array_bytes_size));
  CUDA_CHECK(cudaMemcpy(da, ha.data(), array_bytes_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb.data(), array_bytes_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, hc.data(), array_bytes_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaEventRecord(t1));
  vectorAdd<<<grid_size, block_size>>>(da, db, dc, N);
  CUDA_CHECK(cudaEventRecord(t2));
  CUDA_CHECK(cudaMemcpy(hc.data(), dc, array_bytes_size, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(t3));
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  float t_io_h2d = 0;
  float t_kernel_exec = 0;
  float t_io_d2h = 0;
  float t_total = 0;
  cudaEventElapsedTime(&t_io_h2d, t0, t1);
  cudaEventElapsedTime(&t_kernel_exec, t1, t2);
  cudaEventElapsedTime(&t_io_d2h, t2, t3);
  cudaEventElapsedTime(&t_total, t0, t3);
  std::cout << "t_io_h2d time (CUDA events): " << t_io_h2d << " ms" << std::endl;
  std::cout << "Kernel execution time (CUDA events): " << t_kernel_exec << " ms" << std::endl;
  std::cout << "t_io_d2h time (CUDA events): " << t_io_d2h << " ms" << std::endl;
  std::cout << "t_total time (CUDA events): " << t_total << " ms" << std::endl;
  return 0;
}

/*
t_io_h2d time (CUDA events): 0.139264 ms
Kernel execution time (CUDA events): 0.077824 ms
t_io_d2h time (CUDA events): 0.014464 ms
t_total time (CUDA events): 0.231552 ms
 */