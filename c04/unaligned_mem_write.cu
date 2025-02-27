//
// Created by Gary27 on 2025/2/27.
//


#include <iostream>

__global__ void vec_add_with_off(float* a, float *b, float *c, const int n, unsigned int off){
    auto idx = threadIdx.x + blockIdx.x * blockDim.x;
    auto idx_off = idx + off;
    if(idx < n) c[idx_off] = a[idx] + b[idx];
}

int main(int argc, char** argv){

    int off = 10;
    if(argc > 1) off = atoi(argv[0]);

    int dev_idx = 0;
    cudaSetDevice(dev_idx);
    cudaDeviceProp prop {};
    cudaGetDeviceProperties(&prop, dev_idx);
    std::cout << "Device name: " << prop.name << "\n";

    unsigned int data_size = 1 << 22;
    unsigned int n_bytes = data_size * sizeof(float );

    auto* hst_mem_A = new float[data_size];
    auto* hst_mem_B = new float[data_size];
//    auto* hst_mem_C = new float[data_size];

    std::fill_n(hst_mem_A, data_size, 1.0);
    std::fill_n(hst_mem_B, data_size, 2.0);

    float *dvc_mem_A, *dvc_mem_B, *dvc_mem_C;
    cudaMalloc(&dvc_mem_A, n_bytes);
    cudaMalloc(&dvc_mem_B, n_bytes);
    cudaMalloc(&dvc_mem_C, n_bytes);

    cudaMemcpy(dvc_mem_A, hst_mem_A, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_mem_B, hst_mem_B, n_bytes, cudaMemcpyHostToDevice);
    // std::fill_n(dvc_mem_A, n_bytes, 1.0); // ?

    dim3 bs {32, 1, 1};
    dim3 grid{(data_size + bs.x - 1) / bs.x, 1, 1};
    vec_add_with_off<<<grid, bs>>>(dvc_mem_A, dvc_mem_B, dvc_mem_C, data_size, off);

    delete []hst_mem_A;
    delete []hst_mem_B;
    cudaFree(dvc_mem_A);
    cudaFree(dvc_mem_B);
    cudaFree(dvc_mem_C);
    cudaDeviceReset();
    return 0;
}

// gst_efficiency
//
// smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct