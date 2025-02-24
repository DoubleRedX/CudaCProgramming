//
// Created by Gary27 on 2025/2/21.
//

#include <cmdline.h>
#include <iostream>


__global__ void read_offset(float* A, float* B,  float* C, unsigned int n, unsigned int offset){

    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto k = i + offset;
    if(k < n) C[i] = A[k] + B[k];

}


int main(int argc, char** argv){

//    cmdline::parser parser;
//    parser.add<int>("offset", 'o', "offset address", false, 0);
//    auto offset = parser.get<int>("offset");

    int dev_idx = 1;
    cudaSetDevice(dev_idx);
    cudaDeviceProp prop {};
    cudaGetDeviceProperties(&prop, dev_idx);
    std::cout << "Device name: " << prop.name << "\n";
    unsigned int data_size = 1 << 22;
    auto data_bytes = data_size * sizeof(float);

    unsigned int block_size = 512;
    unsigned int offset = 0;

    if(argc > 1) offset = atoi(argv[1]);
    if(argc > 2) block_size = atoi(argv[2]);

    auto *hst_mem_A = new float[data_size];
    auto *hst_mem_B = new float[data_size];
    auto *hst_mem_C = new float[data_size];
    std::fill_n(hst_mem_A, data_size, 1.0);
    std::fill_n(hst_mem_B, data_size, 2.0);

    float *dvc_mem_A, *dvc_mem_B, *dvc_mem_C;
    cudaMalloc(&dvc_mem_A, data_bytes);
    cudaMalloc(&dvc_mem_B, data_bytes);
    cudaMalloc(&dvc_mem_C, data_bytes);

    cudaMemcpy(dvc_mem_A, hst_mem_A, data_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dvc_mem_B, hst_mem_B, data_bytes, cudaMemcpyHostToDevice);

    // kernel 1
    dim3 BS{block_size, 1, 1};
    dim3 grid{(BS.x + data_size - 1)/BS.x, 1, 1};

    read_offset<<<grid, BS>>>(dvc_mem_A, dvc_mem_B, dvc_mem_C, data_size, offset);
    cudaDeviceSynchronize();

    cudaMemcpy(hst_mem_C, dvc_mem_C, data_bytes, cudaMemcpyDeviceToHost);

    std::cout << "hst_mem_C: " << hst_mem_C[0] << "\n";

    cudaFree(dvc_mem_A);
    cudaFree(dvc_mem_B);
    cudaFree(dvc_mem_C);
    delete[] hst_mem_A;
    delete[] hst_mem_B;
    delete[] hst_mem_C;
    cudaDeviceReset();

    return 0;
}
