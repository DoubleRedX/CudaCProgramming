//
// Created by Gary27 on 2025/1/20.
//
#include <cstdint>
#include <cstdio>

__device__ uint64_t shr(uint64_t value, uint32_t shift){
    // Shift a right by the amount specified by unsigned 32-bit value in b.
    uint64_t res;
    asm("shr.b64 %0, %1, %2;" : "=l"(res) : "l"(value), "r"(shift));
    return res;
}

__global__ void test_shr(){
    uint64_t value = 0x12345678FFFFFFFF;
    uint64_t res;
    uint32_t shift = 4 * 2;
    res = shr(value, 64 - shift);

    printf("original value: %lX\n", value);
    printf("after shl value: %lX\n", res);
}


int main(){
    test_shr<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}