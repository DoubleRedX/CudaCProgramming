//
// Created by Gary27 on 2025/7/29.
//

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


__global__ void shuffle_broadcast_kernel(const int* in, int* out) {
    auto g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[g_tid];
    auto bc_val = __shfl_sync(__activemask(), val, 0, 16);
    out[g_tid] = bc_val;
}

__global__ void shuffle_up_broadcast_kernel(const int* in, int* out) {
    auto g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[g_tid];
    auto bc_val = __shfl_up_sync(__activemask(), val, 2, 16);
    out[g_tid] = bc_val;
}

__global__ void shuffle_down_broadcast_kernel(const int* in, int* out) {
    auto g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int val = in[g_tid];
    auto bc_val = __shfl_down_sync(__activemask(), val, 2, 16);
    out[g_tid] = bc_val;
}

int main() {
    constexpr int N = 64;
    thrust::device_vector<int> x(N);
    thrust::device_vector<int> y(N);
    // thrust::fill(x.begin(), x.end(), 9);
    thrust::sequence(x.begin(), x.end());
    dim3 block_size {32, 1, 1};
    dim3 grid_size {(N + block_size.x - 1) / block_size.x, 1, 1};
    // shuffle_broadcast_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(x.data()),thrust::raw_pointer_cast(y.data()));
    // shuffle_up_broadcast_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(x.data()),thrust::raw_pointer_cast(y.data()));
    shuffle_down_broadcast_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(x.data()),thrust::raw_pointer_cast(y.data()));

    thrust::host_vector<int> h_output = y;

    for (int i = 0; i < x.size(); i++) {
        std::cout << "x[" << i << "] = " << x[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < x.size(); i++) {
        std::cout << "y[" << i << "] = " << y[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}