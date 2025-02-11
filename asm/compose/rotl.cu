//
// Created by Gary27 on 2025/1/20.
//
#include <cstdint>
#include <cstdio>

__device__ uint64_t rotl64(uint64_t value, uint32_t shift){
    uint64_t res;
    asm("{\n\t"
        ".reg .b64 lhs;\n\t"
        ".reg .u32 roff;\n\t"
        "shl.b64 lhs, %1, %2;\n\t"
        "sub.u32 roff, 64, %2;\n\t"
        "shr.b64 %0, %1, roff;\n\t"
        "add.u64 %0, lhs, %0;\n\t"
        "}\n" : "=l"(res) : "l"(value), "r"(shift));

    return res;
}

__global__ void test_rotl64(){
    uint64_t value = 0x12345678FFFFFFFF;
    uint32_t shift = 4 * 2;
    auto res = rotl64(value, shift);
    printf("\nvalue 1: %lX", value);
    printf("\nvalue 2: %lX", res);
}


int main(){

    test_rotl64<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}