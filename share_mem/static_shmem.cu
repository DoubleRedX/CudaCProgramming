//
// Created by Gary27 on 2025/7/9.
//


#include <iostream>

#define BDIMX 32
#define BDIMY 32

template<typename T>
__global__ void setRowReadColDyn(T* out) {
    extern __shared__ int tile[];

    unsigned int row_idx = threadIdx.x + blockDim.x * threadIdx.y;
    unsigned int col_idx = threadIdx.y + blockDim.y * threadIdx.x;

    tile[row_idx] = row_idx;

    __syncthreads();

    out[col_idx] = tile[row_idx];
}

int main() {

    int *y_device;
    auto y_host = new int[BDIMX * BDIMY];
    cudaMalloc(&y_device, BDIMX * BDIMY * sizeof(int));
    setRowReadColDyn<<<1, dim3(BDIMX, BDIMY), BDIMX * BDIMY * sizeof(int)>>>(y_device);
    cudaDeviceSynchronize();
    cudaMemcpy(y_host, y_device, BDIMX * BDIMY * sizeof(int), cudaMemcpyDeviceToHost);  // dynamic share memory 需要在launch期间指定share memory的大小

    std::cout << "16, 16: " << y_host[16 + 16 * 32] << std::endl;
    cudaFree(y_device);
    delete [] y_host;

    return 0;
}