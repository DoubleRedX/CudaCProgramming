//
// Created by Gary27 on 2025/3/17.
//
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>


/**
 * @brief 其他类型转无符号8位整数
 * @tparam T
 *
 *
 * @param u
 * @return
 */
template<typename T>
__device__ __forceinline__ unsigned char t2unsigned_char(T u){
    if(u < 0) return 0;
    if (u > 255) return 255;
    auto out = static_cast<unsigned char>(u);
    return out;
}

__device__ __forceinline__ unsigned char t2u8_rnd_float(float u){
    float rnd_value = std::round(u);
    if(rnd_value < 0.0f) return 0;
    if (rnd_value > 255.0f) return 255;
    auto out = static_cast<unsigned char>(rnd_value);
    return out;
}

__device__ __forceinline__ unsigned char t2u8_rnd_double(double u){
    double rnd_value = std::round(u);
    if(rnd_value < 0.0) return 0;
    if (rnd_value > 255.0) return 255;
    auto out = static_cast<unsigned char>(rnd_value);
    return out;
}

///////////////////////////// 无符号8位

/**
 * @brief 有符号整数转无符号整数，并引用饱和处理
 */
__global__ void test_cvt_sat_u8_s8() {
    signed char input = 127; // 输入值为负数，超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u8.s8 %0, %1;" : "=r"(out) : "r"(static_cast<int>(input)));

    printf("Result of cvt.sat.u8.s8: %u\n", out);

    auto res2 = t2unsigned_char<signed char>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

/**
 * @brief 有符号16位整数转无符号8位整数，并应用饱和处理
 */
__global__ void test_cvt_sat_u8_s16() {
    short input = 90; // 输入值为负数，超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u8.s16 %0, %1;" : "=r"(out) : "h"(input));

    printf("Result of cvt.sat.u8.s16: %u\n", out);
}

/**
 * @brief 无符号16位整数转无符号8位整数，并应用饱和处理
 */
__global__ void test_cvt_sat_u8_u16() {
    unsigned short input = 300; // 输入值超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u8.u16 %0, %1;" : "=r"(out) : "h"(input));

    // 输出结果
    printf("Result of cvt.sat.u8.u16: %u\n", out);
}

/**
 * @brief 有符号32位整数转无符号8位整数，应用饱和处理
 */

__global__ void test_cvt_sat_u8_s32() {
    int input = 255; // 输入值为负数，超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u8.s32 %0, %1;" : "=r"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.u8.s32: %u\n", out);
}

/**
 * @brief 无符号32位整数转无符号8位整数，应用饱和处理
 */
__global__ void test_cvt_sat_u8_u32() {
    unsigned int input = 400; // 输入值超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u8.u32 %0, %1;" : "=r"(out) : "r"(input));

    printf("Result of cvt.sat.u8.u32: %u\n", out);
}

__global__ void test_saturate_cast() {
    float input = 300.0f; // 输入值超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    // rni: 四舍五入到最近的整数值
    // sat: 如果结果超出目标类型的范围，则将其限制在范围内。
    asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(input));

    printf("Result of cvt.rni.sat.u8.f32: %u\n", out);

    auto res2 = t2u8_rnd_float(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

/**
 * @brief 双精度浮点数转无符号8位整数，并应用四舍五入到最近的整数RNI和饱和处理SAT
 */
__global__ void test_cvt_rni_sat_u8_f64() {
    double input = 99.9; // 输入值为负数，超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(out) : "d"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.u8.f64: %u\n", out);

    auto res2 = t2u8_rnd_double(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

///////////////////////////////////////// 有符号8位

/**
 * @brief 无符号8位整数转有符号8位整数，并应用饱和处理。
 */
__global__ void test_cvt_sat_s8_u8() {
    unsigned char input = 200; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s8.u8 %0, %1;" : "=r"(out) : "r"(static_cast<unsigned int>(input)));

    // 输出结果
    printf("Result of cvt.sat.s8.u8: %d\n", static_cast<signed char>(out));
}



/**
 * @brief
 * @return
 */
int main() {

    std::cout << "Call ASM: cvt.sat.u8.s8" << std::endl;
    test_cvt_sat_u8_s8<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Call ASM: cvt.sat.u8.s16" << std::endl;
    test_cvt_sat_u8_s16<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Call ASM: cvt.sat.u8.u16" << std::endl;
    test_cvt_sat_u8_u16<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Call ASM: cvt.sat.u8.s32" << std::endl;
    test_cvt_sat_u8_s32<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Call ASM: cvt.sat.u8.u32" << std::endl;
    test_cvt_sat_u8_u32<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Call ASM: cvt.rni.sat.u8.f32" << std::endl;
    test_saturate_cast<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "Call ASM: cvt.rni.sat.u8.f64" << std::endl;
    test_cvt_rni_sat_u8_f64<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "------------------------------s8------------------------------" << std::endl;

    std::cout << "Call ASM: cvt.sat.s8.u8" << std::endl;
    test_cvt_sat_s8_u8<<<1, 1>>>();
    cudaDeviceSynchronize();


    return 0;
}