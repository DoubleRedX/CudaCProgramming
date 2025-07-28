//
// Created by Gary27 on 2025/7/8.
//

#include <iostream>
#include <numeric>
#include <vector>

template<typename T, int N>
void initialize_vec_host(std::vector<T>& vec_host) {
    for (int i = 0; i < N; i++) vec_host[i] = i;
}

template<typename T>
__global__ void reduce_sum(T *in, T *out, int len) {

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int s = 1; tid + s < len; s*=2) {                  // tid + s < len: assume tid == 0,
        if (tid % (s * 2) == 0) in[tid] += in[tid + s];
        __syncthreads();    // 不是跨块同步的话这怎么起作用的？
    }

    if (tid == 0) *out = in[0];
}

template<typename T, int BlockSize>
__global__ void reduce_sum_share(T *in, T *out, int len) {  // thread > data.x的时候需要处理一下
    __shared__ T tile[BlockSize];
    unsigned int g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tid = threadIdx.x;

    if (g_tid >= len) return;
    tile[tid] = in[g_tid];

    __syncthreads();

    for (int s = 1; tid + s < blockDim.x; s*=2) {
        if (tid % (2*s) == 0) tile[tid] += tile[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = tile[0];
}

#define BLOCK_SIZE_X 32

void reduce_v2() {
    const int n = 1024;
    unsigned int out_size = n / BLOCK_SIZE_X;

    auto x_host = new int[n];
    auto y_host = new int[out_size];
    int *x_device, *y_device;

    for (int i=0;i<n;i++) x_host[i] = i;
    for (int i = 0; i < out_size; ++i) y_host[i] = 0;

    dim3 block_size {BLOCK_SIZE_X, 1, 1};
    dim3 grid_size {(n + BLOCK_SIZE_X -1) / BLOCK_SIZE_X, 1, 1};

    cudaMalloc(&x_device, sizeof(int) * n);
    cudaMalloc(&y_device, sizeof(int) * out_size);

    cudaMemcpy(x_device, x_host, sizeof(int) * n, cudaMemcpyHostToDevice);
    reduce_sum_share<int, BLOCK_SIZE_X><<<grid_size, block_size>>>(x_device, y_device, n);

    cudaMemcpy(y_host, y_device, sizeof(int) * out_size, cudaMemcpyDeviceToHost);

    auto sum = std::accumulate(y_host, y_host + out_size, 0);

    std::cout << "sum = " << sum << std::endl;
    cudaFree(x_device);
    cudaFree(y_device);
    delete[] x_host;
    delete[] y_host;
}

void reduce_v1() {
    const int N = 1024;

    std::vector<int> x_host(N);
    initialize_vec_host<int, N>(x_host);

    auto y_host = std::accumulate(x_host.begin(), x_host.end(), 0);
    std::cout << "host sum = " << y_host << std::endl;

    int* x_device, *y_device;
    dim3 block_size {32, 1, 1};
    dim3 grid_size {(N + 32 - 1) / 32, 1, 1};

    cudaMalloc(&x_device, sizeof(int) * N);
    cudaMalloc(&y_device, sizeof(int));
    int y_device_on_host = 0;
    cudaMemcpy(x_device, x_host.data(), sizeof(int)*N, cudaMemcpyHostToDevice);

    reduce_sum<<<grid_size, block_size>>>(x_device, y_device, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&y_device_on_host, y_device, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "y_device_on_host = " << y_device_on_host << std::endl;
    cudaFree(x_device);
    cudaFree(y_device);
    delete[] x_device;

}



// 单个block能够启动的最大warp数如何计算？

int main() {

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "max warps per block: " << prop.maxThreadsPerBlock / 32 << std::endl;

    std::cout << "call reduce_v1..." << std::endl;
    reduce_v1();


    std::cout << "call reduce_v2..." << std::endl;
    reduce_v2();


    return 0;
}