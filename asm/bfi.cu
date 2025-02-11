//
// Created by Gary27 on 2025/1/16.
//


#include <cuda_runtime.h>
#include <iostream>

// bfi.b64 f, a, b, c, d
// insert a to b at position c with len d, save res to f.
__device__ uint64_t bfi64(uint64_t src, uint64_t dst, uint32_t position) {
    // insert value 2 src and save in dst
    auto result = src;
    asm("bfi.b64 %0, %1, %2, %3, 8;" : "=l"(result) : "l"(src), "l"(dst), "r"(position << 3));
    return result;
}

__device__ void set_byte_bfi64(uint64_t& a, uint32_t position, uint64_t value){
    asm("bfi.b64 %0,%1,%2,%3,8;" : "=l"(a) : "l"(value), "l"(a), "r"(position << 3));
}

__device__ uint32_t bfi32(uint32_t src, uint32_t dst, uint32_t position) {
    auto result = src; // 基础值
    asm("bfi.b32 %0, %1, %2, %3, 8;" : "=r"(result) : "r"(src), "r"(dst), "r"(position << 3)); // 插入位段
    // bfi.b64 dest, src, base, offset, width;
    return result;
}

//__device__ uint64_t bfi64_v1(uint64_t src, uint64_t dst, uint32_t position, uint64_t len) {
//    // 其中position和len的长度只能是r
//    auto result = src; // 基础值
//    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=l"(result) : "l"(src), "l"(dst), "r"(position), "l"(len)); // 插入位段
//    // bfi.b64 dest, src, base, offset, width;
//    return result;
//}

__global__ void test_bfi() {
    uint64_t src = 0x00F0F021;
    uint64_t dst = 0xFFFFFFFF00000000;
    uint32_t position = 0;  // 第k个字节  // 7 6 5 4 3 2 1 0 //

    auto res64 = bfi64(src, dst, position);
    printf("res64: %lX\n", res64);


    uint64_t src2 = 0x00F0F021;
    uint64_t dst2 = 0xFFFFFFFF00000000;
    uint32_t position2 = 0;
    printf("dst2 before: %lX\n", dst2);
    set_byte_bfi64(dst2, position2, src2);
    printf("dst2 after: %lX\n", dst2);


//    uint64_t src64_2 = 0x0000000F;
//    uint64_t dst64_2 = 0xFFFFFFFF00000000;
//    uint32_t position_v1 = 0;  // 第k个字节  // 7 6 5 4 3 2 1 0 //
//    uint64_t len_v1 = 4;
//
//    auto res64_2 = bfi64_v1(src64_2, dst64_2, position_v1, len_v1);
//    printf("res64_v1: %lX\n", res64_2);

    uint32_t src32 = 0x0000000F;
    uint32_t dst32 = 0xFFFF0000;
    uint32_t position32 = 2;  // 第k个字节

    auto res32= bfi32(src32, dst32, position32);
    printf("res32: %X", res32);
}



int main() {
    test_bfi<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}