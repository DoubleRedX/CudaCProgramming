//
// Created by Gary27 on 2025/3/25.
//


#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>


#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUSOLVER_ERROR(err) \
    if (err != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSolver error: " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // 初始化参数
    cusolverDnHandle_t handle;
    syevjInfo_t params;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // 计算特征值和特征向量
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;    // 下三角部分有效
    const int n = 3;                                   // 矩阵维度
    const int batch_count = 2;                         // 批量大小
    const int lda = n;                                 // 矩阵的leading dimension

    // 创建句柄和参数对象
    CHECK_CUSOLVER_ERROR(cusolverDnCreate(&handle));
    CHECK_CUSOLVER_ERROR(cusolverDnCreateSyevjInfo(&params));

    // 设置参数（可选）
    CHECK_CUSOLVER_ERROR(cusolverDnXsyevjSetTolerance(params, 1.0e-7));
    CHECK_CUSOLVER_ERROR(cusolverDnXsyevjSetMaxSweeps(params, 100));

    // 主机上的输入数据：两个3x3对称矩阵
    std::vector<float> A_host = {
        // 第一个矩阵（对称）
        2.0f, -1.0f, 0.0f,
        -1.0f, 2.0f, -1.0f,
        0.0f, -1.0f, 2.0f,
        // 第二个矩阵（对称）
        1.0f, 0.5f, 0.0f,
        0.5f, 1.0f, 0.5f,
        0.0f, 0.5f, 1.0f
    };

    // 设备内存分配
    float *d_A, *d_W;
    int *d_info;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeof(float) * n * n * batch_count));
    CHECK_CUDA_ERROR(cudaMalloc(&d_W, sizeof(float) * n * batch_count));
    CHECK_CUDA_ERROR(cudaMalloc(&d_info, sizeof(int) * batch_count));

    // 将数据复制到设备
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A_host.data(), sizeof(float) * n * n * batch_count, cudaMemcpyHostToDevice));

    // 步骤1：查询工作缓冲区大小
    int lwork;
    CHECK_CUSOLVER_ERROR(cusolverDnSsyevjBatched_bufferSize(
        handle, jobz, uplo, n, d_A, lda, d_W, &lwork, params, batch_count));

    // 分配工作缓冲区
    float *d_work;
    CHECK_CUDA_ERROR(cudaMalloc(&d_work, sizeof(float) * lwork));

    // 步骤2：计算特征值和特征向量
    CHECK_CUSOLVER_ERROR(cusolverDnSsyevjBatched(
        handle, jobz, uplo, n, d_A, lda, d_W, d_work, lwork, d_info, params, batch_count));

    // 将结果复制回主机
    std::vector<float> W_host(n * batch_count);
    std::vector<float> A_result(n * n * batch_count);
    std::vector<int> info_host(batch_count);

    CHECK_CUDA_ERROR(cudaMemcpy(W_host.data(), d_W, sizeof(float) * n * batch_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(A_result.data(), d_A, sizeof(float) * n * n * batch_count, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(info_host.data(), d_info, sizeof(int) * batch_count, cudaMemcpyDeviceToHost));

    // 打印结果
    for (int b = 0; b < batch_count; ++b) {
        std::cout << "Batch " << b << ":\n";
        std::cout << "Info: " << info_host[b] << "\n";

        std::cout << "Eigenvalues:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << W_host[b * n + i] << " ";
        }
        std::cout << "\n";

        if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
            std::cout << "Eigenvectors (column major):\n";
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    std::cout << A_result[b * n * n + i + j * n] << " ";
                }
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

    // 释放资源
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_W));
    CHECK_CUDA_ERROR(cudaFree(d_info));
    CHECK_CUDA_ERROR(cudaFree(d_work));
    CHECK_CUSOLVER_ERROR(cusolverDnDestroySyevjInfo(params));
    CHECK_CUSOLVER_ERROR(cusolverDnDestroy(handle));

    return 0;
}