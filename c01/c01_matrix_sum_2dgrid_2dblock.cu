//
// Created by Gary27 on 2024/12/26.
//


#include <vector>
#include <helper.h>

#include "aux.h"

__global__ void matrix_sum_2d_grid_2d_block(float *a, float *b, float *c, int m, int n) {
    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x < m && y < n) {
        auto idx = x + y * n;
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrix_sum_1d_grid_1d_block(float *a, float *b, float *c, int m, int n) {
    // 这个核函数，直接跳过矩阵的抽象，视矩阵为一维的向量
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < m * n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrix_sum_1d_grid_1d_block_v2(float *a, float *b, float *c, int m, int n) {
    // 该矩阵按照矩阵的列数n展开，每个thread处理一列数据
    auto mj = threadIdx.x + blockIdx.x * blockDim.x;
    if (mj < n) {
        for (unsigned int i = 0; i < m; ++i) {
            auto mi = i;
            auto idx = mj + mi * n;
            c[idx] = a[idx] + b[idx];
        }
    }
}

__global__ void matrix_sum_2d_grid_1d_block(float *a, float *b, float *c, int m, int n) {
    // matrix sum with 2d grid and 1d block
    auto mj = threadIdx.x + blockIdx.x * blockDim.x;
    auto mi = blockIdx.y;
    if (mj < n) {
        auto idx = mi + mj * n;
        c[idx] = a[idx] + b[idx];
    }
}


int main() {
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Total registers per block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Maximum threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Maximum threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;

    cudaSetDevice(device);
    constexpr int m = 10000;
    constexpr int n = 8000;

    std::vector<float> hma(m * n); // std::vector<float> hma(m * n, 0.0); // 按行优先存储
    std::vector<float> hmb(m * n);
    std::vector<float> hmc(m * n);
    // std::array<std::array<float, n>, m> hmc;  // array 在栈上分配，如果弄太大会栈溢出也就是exit code 139

    size_t matrix_size_in_bytes = m * n * sizeof(float);
    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));
    CUDA_CHECK(cudaEventCreate(&t3));

    try {
        initializeMatrix<float>(hma, 1.);
        initializeMatrix<float>(hmb, 2.);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    float *dma, *dmb, *dmc;
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMalloc(&dma, matrix_size_in_bytes));
    CUDA_CHECK(cudaMalloc(&dmb, matrix_size_in_bytes));
    CUDA_CHECK(cudaMalloc(&dmc, matrix_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(dma, hma.data(), matrix_size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmb, hmb.data(), matrix_size_in_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    dim3 block_size{32, 32};
    dim3 grid_size{(m + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y};
    matrix_sum_2d_grid_2d_block<<<grid_size, block_size>>>(dma, dmb, dmc, m, n);
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaMemcpy(hmc.data(), dmc, matrix_size_in_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t3));
    float time_io_h2d{}, time_kernel{}, time_io_d2h{};
    CUDA_CHECK(cudaEventElapsedTime(&time_io_h2d, t0, t1));
    CUDA_CHECK(cudaEventElapsedTime(&time_kernel, t1, t2));
    CUDA_CHECK(cudaEventElapsedTime(&time_io_d2h, t2, t3));
    CUDA_CHECK(cudaFree(dma));
    CUDA_CHECK(cudaFree(dmb));
    CUDA_CHECK(cudaFree(dmc));
    cudaDeviceReset();
    std::cout << "time_io_h2d: " << time_io_h2d << " ms" << std::endl;
    std::cout << "time_kernel: " << time_kernel << " ms" << std::endl;
    std::cout << "time_io_d2h: " << time_io_d2h << " ms" << std::endl;
    return 0;
}
