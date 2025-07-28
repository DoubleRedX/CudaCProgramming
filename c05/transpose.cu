//
// Created by Gary27 on 2025/7/16.
//

#include <iostream>
#include <numeric>
#include <vector>

#define INDEX(iCOL, iROW, STRIDE) (iCOL + iROW * STRIDE)
#define WARP_SIZE 32

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

__global__ void naiveGmem(int *out, int *in, const int nrows, const int ncols) {
    // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // if (idx < ncols && idy < nrows) {
    //     out[INDEX(idx, idy, nrows)] = in[INDEX(idy, idx, ncols)];
    // }

    auto coord_x = threadIdx.x + blockDim.x * blockIdx.x; // row-major
    auto coord_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (coord_x < ncols && coord_y < nrows) {
        out[INDEX(coord_y, coord_x, nrows)] = in[INDEX(coord_x, coord_y, ncols)];
    }
}

__global__ void copyGmem(int *out, int *in, const int nrows, const int ncols) {
    auto coord_x = threadIdx.x + blockDim.x * blockIdx.x;
    auto coord_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (coord_x < ncols && coord_y < nrows) {
        out[INDEX(coord_x, coord_y, ncols)] = in[INDEX(coord_x, coord_y, ncols)];
    }
}


template<typename T>
__global__ void transposeSmem(T *out, T *in, const int nrows, const int ncols) {
    __shared__ T tile[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    auto g_tid_x = threadIdx.x + blockDim.x * blockIdx.x;
    auto g_tid_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (g_tid_x < ncols && g_tid_y < nrows) {
        tile[threadIdx.y][threadIdx.x] = in[INDEX(g_tid_x, g_tid_y, ncols)];
    }

    __syncthreads();

    auto b_tid = threadIdx.x + threadIdx.y * blockDim.x;
    auto t_b_tid_x = b_tid % blockDim.y;
    auto t_b_tid_y = b_tid / blockDim.y;

    printf("(%d, %d)----(%d, %d)----(%d, %d)\n", g_tid_x, g_tid_y, threadIdx.x, threadIdx.y, t_b_tid_x, t_b_tid_y);

    auto t_g_tid_x = t_b_tid_x + blockDim.y * blockIdx.y;
    auto t_g_tid_y = t_b_tid_y + blockDim.x * blockIdx.x;

    __syncthreads();
    if (t_g_tid_x < nrows && t_g_tid_y < ncols) {
        out[INDEX(t_g_tid_x, t_g_tid_y, nrows)] = tile[t_b_tid_x][t_b_tid_y]; // share 读的顺序需要参考share memory的形状
    }

    // if (g_tid_x < ncols && g_tid_y < nrows) {
        // out[INDEX(g_tid_y, g_tid_x, nrows)] = tile[t_b_tid_x][t_b_tid_y];            // 为什么这种写法不对
    // }
}

template<typename T>
__global__ void transposeSmemUnroll(T* out, T* in, const int nrows, const int ncols) {
    /*
     * 使用循环展开的时候要记住：启动配置也要修改
     */
    __shared__ T tile[BLOCK_SIZE_Y][BLOCK_SIZE_X * 2];

    auto g_tid_x = threadIdx.x + blockDim.x * blockIdx.x * 2;
    auto g_tid_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (g_tid_x < ncols && g_tid_y < nrows) {
        tile[threadIdx.y][threadIdx.x] = in[INDEX(g_tid_x, g_tid_y, ncols)];
    }
    if (g_tid_x + blockDim.x < ncols && g_tid_y < nrows) {
        tile[threadIdx.y][threadIdx.x + blockDim.x] = in[INDEX(g_tid_x+blockDim.x, g_tid_y, ncols)];
    }
    __syncthreads();

    auto b_tid = threadIdx.x + threadIdx.y * blockDim.x;
    auto t_b_tid_x = b_tid % blockDim.y;
    auto t_b_tid_y = b_tid / blockDim.y;
    auto t_g_tid_x = t_b_tid_x + blockDim.y * blockIdx.y;
    auto t_g_tid_y = t_b_tid_y + 2 * blockDim.x * blockIdx.x;

    if (t_g_tid_x < nrows && t_g_tid_y < ncols) {
        out[INDEX(t_g_tid_x, t_g_tid_y, nrows)] = tile[t_b_tid_x][t_b_tid_y];
    }
    if (t_g_tid_x < nrows && t_g_tid_y + blockDim.x < ncols) {
        out[INDEX(t_g_tid_x, t_g_tid_y + blockDim.x, nrows)] = tile[t_b_tid_x][t_b_tid_y + blockDim.x];
    }

}


template<typename T>
__global__ void transposeSmemUnrollV2(T *out, T *in, const int nrows, const int ncols)
{
    // static 1D shared memory
    __shared__ float tile[BLOCK_SIZE_Y][BLOCK_SIZE_X * 2];

    // coordinate in original matrix
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x;

    // linear global memory index for original matrix
    unsigned int offset = INDEX(col, row, ncols);
    unsigned int offset2 = INDEX(col2, row2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // linear global memory index for transposed matrix
    unsigned int transposed_offset = INDEX(row, col, nrows);
    unsigned int transposed_offset2 = INDEX(row2, col2, nrows);

    if (row < nrows && col < ncols)
    {
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols)
    {
        out[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols)
    {
        out[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}


template<typename T>
void print(const std::vector<T> &vec, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << vec[j + i * n] << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void initialize_vec(std::vector<T> &vec) {
    std::iota(vec.begin(), vec.end(), 1);
}

int main() {
    const int M = 64;
    const int N = 128;
    // const int M = 4096;
    // const int N = 4096;

    std::vector<int> x(M * N);
    std::vector<int> y(M * N);
    // auto x = new int[M*N];
    // auto y = new int[M*N];
    std::iota(x.begin(), x.end(), 1);
    // std::iota(x, x + M*N, 1);


    int *x_device = nullptr, *y_device = nullptr;
    cudaMalloc(&x_device, M * N * sizeof(int));
    cudaMemcpy(x_device, x.data(), M * N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&y_device, M * N * sizeof(int));

    // dim3 block_size(WARP_SIZE, WARP_SIZE, 1);
    dim3 block_size(32, 16, 1);

    transposeSmemUnroll<<<dim3(N + block_size.x * 2 - 1 / (block_size.x * 2), M + block_size.y - 1 / block_size.y, 1), block_size>>>(
        y_device, x_device, M, N);

    cudaMemcpy(y.data(), y_device, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    print(y, N, M);
    cudaFree(x_device);
    cudaFree(y_device);
    return 0;
}

// ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio, l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio ./app
// ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio ./app

// shared_load_throughput                                           l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second
// shared_load_transactions                                         l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
// shared_load_transactions_per_request                             n/a

// shared_store_throughput                                          l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second
// shared_store_transactions                                        l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
// shared_store_transactions_per_request                            n/a

// gld_transactions                                                 l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
// gld_transactions_per_request                                     l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio