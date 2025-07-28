//
// Created by Gary27 on 2025/7/15.
//

#define DIM 128
#include <iostream>


///*
__global__ void blockExpandKernel(int *g_idata, int *g_odata, unsigned int n) {
    __shared__ int tile[32];

    auto idx = threadIdx.x + 4 * blockDim.x * blockIdx.x;   // data index: thread index to the first data block
    auto tid = threadIdx.x;                                 // thread index: thread index in block
    // auto* idata = g_idata + blockDim.x * blockIdx.x;
    int tmpSum = 0;
    // if (idx + blockDim.x * 3 < n) {  // 这种写法是80, 当然如果是整齐的数据，那么这种方式显然更快，少了if的分支
        // int e1 = g_idata[idx];
        // int e2 = g_idata[idx + blockDim.x * 1];
        // int e3 = g_idata[idx + blockDim.x * 2];
        // int e4 = g_idata[idx + blockDim.x * 3];
        // tmpSum = e1 + e2 + e3 + e4;
    // }

    if (idx < n) {
        int e1 = g_idata[idx];
        tmpSum += e1;
    }
    if (idx + blockDim.x < n) {
        int e2 = g_idata[idx + blockDim.x * 1];
        tmpSum += e2;
    }
    if (idx + blockDim.x * 2 < n) {
        int e3 = g_idata[idx + blockDim.x * 2];
        tmpSum += e3;
    }
    if (idx + blockDim.x * 3 < n) {
        int e4 = g_idata[idx + blockDim.x * 3];
        tmpSum += e4;
    }

    tile[tid] = tmpSum;

    // __syncthreads();

    if (tid < 16) {
        tile[tid] += tile[tid + 16];    // 这个地方需要考虑执行顺序的问题吗？
        tile[tid] += tile[tid +  8];
        tile[tid] += tile[tid +  4];
        tile[tid] += tile[tid +  2];
        tile[tid] += tile[tid +  1];
    }

    __syncthreads();
    if (tid == 0) g_odata[blockIdx.x] = tile[0];


}
//*/

// __global__ void blockExpandKernel(int *g_idata, int *g_odata, unsigned int n) {
//     __shared__ int tile[32];
//
//     int tid = threadIdx.x;
//     int idx = tid + blockDim.x * blockIdx.x * 4;
//
//     int tmpSum = 0;
//     if (idx + 3 * blockDim.x < n) {
//         tmpSum += g_idata[idx];
//         tmpSum += g_idata[idx + blockDim.x];
//         tmpSum += g_idata[idx + 2 * blockDim.x];
//         tmpSum += g_idata[idx + 3 * blockDim.x];
//     }
//     printf("tmpSum = %d\n", tmpSum);
//
//     tile[tid] = tmpSum;
//
//     __syncthreads();
//
//     // Reduction using first 16 threads
//     if (tid < 16) {
//         tile[tid] += tile[tid + 16];
//         tile[tid] += tile[tid + 8];
//         tile[tid] += tile[tid + 4];
//         tile[tid] += tile[tid + 2];
//         tile[tid] += tile[tid + 1];
//     }
//     __syncthreads();
//
//
//     if (tid == 0) g_odata[blockIdx.x] = tile[0];
// }


/*
int main() {


    int x_host[DIM];
    // std::fill(x_host, x_host + DIM - 12, 1);
    std::fill(x_host, x_host + DIM, 1);
    // std::fill(x_host + DIM - 12, x_host + DIM, 0);

    int* x_device, *y_device;
    int res = 0;
    cudaMalloc(&x_device, DIM * sizeof(int));
    cudaMalloc(&y_device, 1 * sizeof(int));
    cudaMemcpy(x_device, x_host, DIM * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemset(y_device, 0, DIM * sizeof(int) / unroll_factor);

    blockExpandKernel<<<dim3(1,1,1), dim3(32, 1, 1)>>>(x_device, y_device, (DIM));

    cudaMemcpy(&res, y_device, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Res: " << res << std::endl;


    return 0;
}
*/

int main() {
    // 设置输入数据（DIM = 128），后 12 个为 0，其余为 1
    int h_input[DIM];
    std::fill(h_input, h_input + (DIM - 12), 1);   // 前 116 个填 1
    std::fill(h_input + (DIM - 12), h_input + DIM, 0); // 后 12 个填 0

    int *d_input, *d_output;
    int h_result = 0;

    cudaMalloc(&d_input, sizeof(int) * DIM);
    cudaMalloc(&d_output, sizeof(int));  // 这里只需要一个 int 输出

    cudaMemcpy(d_input, h_input, sizeof(int) * DIM, cudaMemcpyHostToDevice);

    // 启动 kernel，1 个 block，BLOCK_SIZE 个线程
    blockExpandKernel<<<1, 32>>>(d_input, d_output, DIM - 12);

    cudaMemcpy(&h_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Result: " << h_result << std::endl;

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}