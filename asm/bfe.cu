//
// Created by Gary27 on 2025/1/16.
//
#include <cstdio>
#include <cstdint>

__device__ uint32_t get_byte(uint64_t a, uint32_t position)
{
    uint64_t result;
    asm("bfe.u64 %0,%1,%2,8;" : "=l"(result) : "l"(a), "r"(position * 8));
    return static_cast<uint32_t>(result);
}

__device__ uint32_t bfe_test(uint32_t a){
    uint32_t res;
    asm("// BFE TEST BEGIN");
    asm("bfe.u32 %0, %1, 21, 5;" : "=r"(res) : "r"(a));
    asm("// BFE TEST BEGIN");
    return res;
}

__global__ void test_kernel() {
    uint64_t a = 0x123456789ABCDEF0; // 64 位整数
    uint32_t byte;

    // 提取第 0 个字节（最低字节）
    byte = get_byte(a, 0); // 结果应为 0xF0
    printf("Byte 0: 0x%02X\n", byte);

    // 提取第 2 个字节
    byte = get_byte(a, 2); // 结果应为 0xBC
    printf("Byte 2: 0x%02X\n", byte);

    // 提取第 7 个字节（最高字节）
    byte = get_byte(a, 7); // 结果应为 0x12
    printf("Byte 7: 0x%02X\n", byte);


    uint32_t a32 = 0x22222222;
    auto res = bfe_test(a32);
    printf("bfe: 0x%X\n", res);
}

int main() {
    test_kernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // 等待内核执行完成
    return 0;
}