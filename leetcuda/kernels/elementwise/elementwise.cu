//
// Created by Gary27 on 2025/7/10.
//

#include <cuda_fp16.hpp>
#include <cuda_runtime.h>

// #define FLOAT4(value) ((reinterpret_cast<const float4*>(&value))[0])
#define FLOAT4(value)           (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value)            (reinterpret_cast<half2  *>(&(value))[0])
#define LDST128BITS(value)      (reinterpret_cast<float4 *>(&(value))[0])


// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32_kernel(float* a, float* b, float* c, int len) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= len) return;
    c[gid] = a[gid] + b[gid];
}

// ElementWise Add + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f32x4_kernel(float* a, float* b, float* c, int len) {
    unsigned int gid_scale_4 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (gid_scale_4 >= len) return;
    float4 reg_a = FLOAT4(a[gid_scale_4]);      // 向量化的优化,
    float4 reg_b = FLOAT4(b[gid_scale_4]);      // 有一个问题，在实际的算子开发中，向量化的使用对于不规则的shape，比如说1024+1，后边这个1reinterpret成4不会导致非法内存访问吗？
    float4 reg_c;

    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[gid_scale_4]) = reg_c;
}

// FP16
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16_kernel(half* a, half* b, half* c, int len) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_f16x2_kernel(half* a, half* b, half* c, int len) {
    unsigned int idx = 2 * (threadIdx.x + blockDim.x * blockIdx.x);
    if (idx < len) {
        const half2 reg_a = HALF2(a[idx]);  // 整到寄存器中了
        const half2 reg_b = HALF2(b[idx]);
        const half2 reg_c = __hadd2(reg_a, reg_b);
        HALF2(c[idx]) = reg_c;
    }
}

__global__ void elementwise_add_f16x4_kernel(half* a, half* b, half* c, int len) {
    unsigned int idx = 4 * (threadIdx.x + blockDim.x * blockIdx.x);
    if (idx < len) {
        const half2 reg_a_0 = HALF2(a[idx + 0]);
        const half2 reg_b_0 = HALF2(b[idx + 0]);
        const half2 reg_c_0 = __hadd2(reg_a_0, reg_b_0);

        const half2 reg_a_1 = HALF2(a[idx + 2]);
        const half2 reg_b_1 = HALF2(b[idx + 2]);
        const half2 reg_c_1 = __hadd2(reg_a_1, reg_b_1);

        HALF2(c[idx + 0]) = reg_c_0;
        HALF2(c[idx + 2]) = reg_c_1;
    }
}


__global__ void elementwise_add_f16x8_kernel(half* a, half* b, half* c, int len) {  // 进之前就check一下是不是8的倍数
    unsigned int idx = 8 * (threadIdx.x + blockDim.x * blockIdx.x);
    if (idx < len) {
        const half2 reg_a_0 = HALF2(a[idx + 0]);
        const half2 reg_b_0 = HALF2(b[idx + 0]);
        const half2 reg_c_0 = __hadd2(reg_a_0, reg_b_0);

        const half2 reg_a_1 = HALF2(a[idx + 2]);
        const half2 reg_b_1 = HALF2(b[idx + 2]);
        const half2 reg_c_1 = __hadd2(reg_a_1, reg_b_1);

        const half2 reg_a_2 = HALF2(a[idx + 4]);
        const half2 reg_b_2 = HALF2(b[idx + 4]);
        const half2 reg_c_2 = __hadd2(reg_a_2, reg_b_2);

        const half2 reg_a_3 = HALF2(a[idx + 6]);
        const half2 reg_b_3 = HALF2(b[idx + 6]);
        const half2 reg_c_3 = __hadd2(reg_a_3, reg_b_3);


        HALF2(c[idx + 0]) = reg_c_0;
        HALF2(c[idx + 2]) = reg_c_1;
        HALF2(c[idx + 4]) = reg_c_2;
        HALF2(c[idx + 6]) = reg_c_3;
    }
}


__global__ void elementwise_add_fp16x8_pack_kernel(half* a, half* b, half* c, int len) {
    unsigned int idx = 8 * (threadIdx.x + blockDim.x * blockIdx.x);
    half pack_a[8], pack_b[8], pack_c[8];
    LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

#pragma unroll
    for (int i=0;i < 8; i+=2) {
        HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    }

    if (idx + 7 < len) {
        LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
    }
}





int main(){



    return 0;
}