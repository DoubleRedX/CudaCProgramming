//
// Created by Gary27 on 2024/12/23.
//

// chapter 2 启动一个cuda核函数

// 1.核函数调用和主机线程是异步的。
// 2.一些CUDA运行时API调用和主机是隐式同步的。例如cudaMemCopy

// __global__：host调用，device运行。返回类型必须为void。
// __host__：device调用，device运行
// __device：缺省默认__device__限定，
// __host__ __device__：device+host均可调用，均可运行

// 核函数验证
// （1）使用主机端定义的函数对比结果。
// （2）使用<<<1,1>>>的execution configuration执行核函数。

#include "aux.h"
#include "helper.h"

__global__ void vecAdd(float *a, float *b, float *c, size_t size){
  auto t_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (t_idx < size) {
    c[t_idx] = a[t_idx] + b[t_idx];
  }
}


int main(){

  const size_t size = 2000;
  const size_t bytes = size * sizeof(float);

  std::vector<float> host_data_a(size);
  std::vector<float> host_data_b(size);
  std::vector<float> host_data_c(size);


  try {
    initializeData(host_data_a);
    initializeData(host_data_b);
  } catch (std::exception &e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  float *device_data_a, *device_data_b, *device_data_c;
  CUDA_CHECK(cudaMalloc(&device_data_a, bytes));
  CUDA_CHECK(cudaMalloc(&device_data_b, bytes));
  CUDA_CHECK(cudaMalloc(&device_data_c, bytes));

  CUDA_CHECK(cudaMemcpy(device_data_a, host_data_a.data(), bytes, cudaMemcpyHostToDevice));
  std::cout << "data a copied to device." << std::endl;
  CUDA_CHECK(cudaMemcpy(device_data_b, host_data_b.data(), bytes, cudaMemcpyHostToDevice));
  std::cout << "data b copied to device." << std::endl;

  std::cout << "start to calculate..." << std::endl;
  unsigned int BLOCK_SIZE = 256;
  unsigned int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  vecAdd<<<GRID_SIZE, BLOCK_SIZE>>>(device_data_a, device_data_b, device_data_c, size);

  CUDA_CHECK(cudaMemcpy(host_data_c.data(), device_data_c, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_data_a));
  CUDA_CHECK(cudaFree(device_data_b));
  CUDA_CHECK(cudaFree(device_data_c));
  return 0;
}
