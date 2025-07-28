//
// Created by Gary27 on 2025/3/28.
//

#include <cuda_runtime.h>
#include <iostream>

int main(){
  cudaStream_t default_stream = cudaStreamDefault;  // 获取默认流
  unsigned int flags;
  if (cudaStreamGetFlags(default_stream, &flags) != cudaSuccess) {
    std::cerr << "Failed to get default_stream flags" << std::endl;
    cudaStreamDestroy(default_stream);
    return 1;
  }

  std::cout << "Stream flags: " << flags << std::endl;
  // if (flags == cudaStreamDefault) {
  //   std::cout << "This is a default stream" << std::endl;
  // } else if (flags == hipStreamNonBlocking) {
  //   std::cout << "This is a non-blocking stream" << std::endl;
  // }



  return 0;
}