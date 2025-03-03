//
// Created by Gary27 on 2025/3/3.
//

#include <iostream>

struct innerStruct {

    float x, y;

};

__global__ void test_kernel(innerStruct *a, innerStruct *b, const int n){
    auto i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n) {
        b[i].x = a[i].x + 10.f;
        b[i].y = a[i].y + 20.f;
    }
}

int main(){
    constexpr int data_size = 1024;
    auto *aos = new innerStruct[data_size];
    auto *hst_mem_gpu_res = new innerStruct[data_size];
    innerStruct initValue {1., 2.};
    std::fill_n(aos, 10, initValue);
    auto data_bytes = sizeof(innerStruct);
    innerStruct *dvc_mem_a = nullptr, *dvc_mem_b = nullptr;
    cudaMalloc(&dvc_mem_a, data_bytes);
    cudaMalloc(&dvc_mem_b, data_bytes);

    cudaMemcpy(dvc_mem_a, aos, data_bytes, cudaMemcpyHostToDevice);

    dim3 bs{64, 1, 1};
    dim3 grid {(bs.x - 1 + data_size) / bs.x, 1, 1};
    test_kernel<<<grid, bs>>>(dvc_mem_a, dvc_mem_b, data_size);

    cudaMemcpy(hst_mem_gpu_res, dvc_mem_b, data_bytes, cudaMemcpyDeviceToHost);

    std::cout << "hst mem gpu res: " << "x: " <<hst_mem_gpu_res[0].x << ", y: " << hst_mem_gpu_res[0].y << "\n";

    delete[] aos;
    cudaFree(dvc_mem_a);
    cudaFree(dvc_mem_b);

    return 0;
}