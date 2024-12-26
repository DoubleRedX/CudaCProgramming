//
// Created by Gary27 on 2024/12/26.
//


#include <vector>
__global__ void matrix_sum_2d_grid_2d_block(float* a, float* b, float* c, int m, int n) {
  auto x = threadIdx.x + blockIdx.x * blockDim.x;
  auto y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < m && y < n) {
    auto idx = x + y * n;
    c[idx] = a[idx] + b[idx];
  }
}


int main(){

  const int m = 10000;
  const int n = 8000;
  std::pmr::vector<std::vector<int>> hma();


  return 0;
}