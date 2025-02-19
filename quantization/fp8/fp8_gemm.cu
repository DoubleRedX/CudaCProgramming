//
// Created by Gary27 on 2025/2/18.
//

#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

typedef uint8_t fp8_e4m3;

__device__ __inline__ fp8_e4m3 float_to_fp8_e4m3(float x, float scale){
    x *= scale;  // 应用缩放因子防止溢出
    x = max(x, -448.0f);
    x = min(x, 448.0f);
    uint32_t u32 = __float_as_uint(x);
    uint8_t sign = (u32 >> 31) & 0x1;
    int exp = ((u32 >> 23) & 0xFF) - 127; // IEEE754 指数
    uint32_t frac = (u32 >> 20) & 0x7;    // 取前3位尾数
    // 组合为 E4M3（1位符号 + 4位指数 + 3位尾数）
    uint8_t fp8 = (sign << 7) | (((exp + 7) & 0xF) << 3) | (frac & 0x7);
    return fp8;
}

__device__ __inline__ float fp8_e4m3_to_float(fp8_e4m3 x, float scale) {
    uint8_t sign = (x >> 7) & 0x1;
    int exp = ((x >> 3) & 0xF) - 7; // 指数偏移
    uint8_t frac = x & 0x7;
    // 重构为 IEEE754 float
    uint32_t u32 = (sign << 31) | ((exp + 127) << 23) | (frac << 20);
    float val = __uint_as_float(u32);
    return val / scale; // 反量化
}

__global__ void fp8_gemm_kernel(
        const fp8_e4m3* __restrict__ A,
        const fp8_e4m3* __restrict__ B,
        float* __restrict__ C,
        int M, int N, int K,
        float scale_a, float scale_b, float scale_c
) {
    auto row = blockIdx.y * blockDim.y + threadIdx.y;
    auto col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = fp8_e4m3_to_float(A[row * K + k], scale_a);
        float b = fp8_e4m3_to_float(B[k * N + col], scale_b);
        sum += a * b;
    }
    C[row * N + col] = sum * scale_c; // 输出缩放
}

void launch_fp8_gemm(
        const float* A_fp32, const float* B_fp32,
        float* C_fp32, int M, int N, int K,
        float scale_a, float scale_b, float scale_c
) {
    // 分配设备内存
    fp8_e4m3 *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(fp8_e4m3));
    cudaMalloc(&d_B, K * N * sizeof(fp8_e4m3));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 将 FP32 量化为 FP8
    std::vector<fp8_e4m3> A_fp8_host(M * K);
    std::vector<fp8_e4m3> B_fp8_host(K * N);
    for (int i = 0; i < M * K; ++i)
        A_fp8_host[i] = float_to_fp8_e4m3(A_fp32[i], scale_a);
    for (int i = 0; i < K * N; ++i)
        B_fp8_host[i] = float_to_fp8_e4m3(B_fp32[i], scale_b);
    cudaMemcpy(d_A, A_fp8_host.data(), ..., cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_fp8_host.data(), ..., cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    fp8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, scale_a, scale_b, scale_c);

    // 拷贝结果回主机
    cudaMemcpy(C_fp32, d_C, ..., cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main(){
    int M = 256, N = 256, K = 256;
    std::vector<float> A(M * K, 0.1f); // 示例输入
    std::vector<float> B(K * N, 0.1f);
    std::vector<float> C(M * N);

    // 计算缩放因子（简化版：根据最大值）
    float max_a = 0.1f, max_b = 0.1f;
    float scale_a = 448.0f / max_a; // 防止溢出
    float scale_b = 448.0f / max_b;
    float scale_c = 1.0f / (scale_a * scale_b);

    launch_fp8_gemm(A.data(), B.data(), C.data(), M, N, K, scale_a, scale_b, scale_c);

    // 验证结果（理想输出应为 0.1*0.1*256 = 2.56）
    printf("C[0] = %f\n", C[0]);
    return 0;
}


