//
// Created by Gary27 on 2025/3/17.
//
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cassert>


/**
 * @brief 其他类型转无符号8位整数
 * @tparam T
 *
 *
 * @param u
 * @return
 */
template<typename T>
__device__ __forceinline__ unsigned char t2u8(T u){
    if constexpr (std::is_unsigned_v<T>) {
        if (u > 255) return 255;
    } else {
        if(u < 0) return 0;
        if (u > 255) return 255;

    }
    return static_cast<unsigned char>(u);
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


/**
 * @brief 转有符号8位整数
 * @tparam T
 * @param u
 * @return
 */
template<typename T>
__device__ __forceinline__ signed char t2s8(T u) {
    if constexpr (std::is_unsigned_v<T>) {
        if (u > 127) return 127;
    } else {
        if (u > 127) return 127;
        if (u < -128) return -128;
    }
    return static_cast<signed char>(u);
}

__device__ __forceinline__ signed char t2s8_rnd_float(float u) {
    float rnd_value = std::round(u);
    if (rnd_value < -128.0f) return -128;
    if (rnd_value > 127.0f) return 127;
    auto out = static_cast<signed char>(rnd_value);
    return out;
}

__device__ __forceinline__ signed char t2s8_rnd_double(double u){
    double rnd_value = std::round(u);
    if(rnd_value < -128.0) return -128;
    if (rnd_value > 127.0) return 127;
    auto out = static_cast<signed char>(rnd_value);
    return out;
}


/**
 * @brief 转无符号16位整数
 * @tparam T
 * @param u
 * @return
 */
template<typename T>
__device__ __forceinline__ unsigned short t2u16(T u) {
    if constexpr (std::is_unsigned_v<T>) {
        if (u > 65535) return 65535;
    } else {
        if (u < 0) return 0;
        if (u > 65535) return 65535;
    }
    return static_cast<unsigned short>(u);
}

__device__ __forceinline__ unsigned short t2u16_rnd_float(float u) {
    float rnd_value = std::round(u);
    if(rnd_value < 0.0f) return 0;
    if (rnd_value > 65535.0f) return 65535;
    auto out = static_cast<unsigned short>(rnd_value);
    return out;
}

__device__ __forceinline__ unsigned short t2u16_rnd_double(double u) {
    double rnd_value = std::round(u);
    if(rnd_value < 0.0) return 0;
    if (rnd_value > 65535.0) return 65535;
    auto out = static_cast<unsigned short>(rnd_value);
    return out;
}


/**
 * @brief 转有符号16位整数
 * @tparam T
 * @param u
 * @return
 */
template<typename T>
__device__ __forceinline__ short t2s16(T u) {
    if (u < -32768) return -32768;
    if (u > 32767) return 32767;
    auto out = static_cast<short>(u);
    return out;
}

__device__ __forceinline__ short t2s16_rnd_float(float u) {
    float rnd_value = std::round(u);
    if (rnd_value < -32768.0f) return -32768;
    if (rnd_value > 32767.0f) return 32767;
    auto out = static_cast<short>(rnd_value);
    return out;
}

__device__ __forceinline__ short t2s16_rnd_double(double u) {
    double rnd_value = std::round(u);
    if (rnd_value < -32768.0) return -32768;
    if (rnd_value > 32767.0) return 32767;
    auto out = static_cast<short>(rnd_value);
    return out;
}

/**
 * @brief 无符号32位整数转有符号32位整数
 */
__device__ __forceinline__ int u32_2_s32(unsigned int u) {
    if (u > 2147483647) return 2147483647;
    auto out = static_cast<int>(u);
    return out;
}

/**
 * @brief 转无符号32位整数
 */
template<typename T>
__device__ __forceinline__ unsigned int t2u32(T u) {
    if (u < 0) return 0;
    if (u > 2147483647) return 2147483647;
    auto out = static_cast<unsigned int>(u);
    return out;
}

///////////////////////////// 无符号8位 ///////////////////////////////////////

/**
 * @brief 有符号整数转无符号整数，并引用饱和处理
 */
__global__ void test_cvt_sat_u8_s8() {
    signed char input = 127; // 输入值为负数，超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u8.s8 %0, %1;" : "=r"(out) : "r"(static_cast<int>(input)));

    printf("Result of cvt.sat.u8.s8: %u\n", out);

    auto res2 = t2u8<signed char>(input);
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
    auto res2 = t2u8<short>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
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
    auto res2 = t2u8<unsigned short>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
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
    auto res2 = t2u8<int>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
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
    auto res2 = t2u8<unsigned int>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_saturate_cast() {
    float input = 120.5f; // 输入值超出 unsigned char 的范围 [0, 255]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    // rni: 四舍五入到最近的整数值
    // sat: 如果结果超出目标类型的范围，则将其限制在范围内。
    asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(input));

    printf("Result of cvt.rni.sat.u8.f32: %u\n", out);

    auto res2 = t2u8_rnd_float(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
    assert(res2 == out);
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

///////////////////////////////////////// 有符号8位 /////////////////////////////////////////////////

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
    auto res2 = t2s8<unsigned char>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


/**
 * @brief 有符号16位整数转有符号8位整数，并应用饱和处理。
 */
__global__ void test_cvt_sat_s8_s16() {
    short input = -200; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s8.s16 %0, %1;" : "=r"(out) : "h"(input));

    // 输出结果
    printf("Result of cvt.sat.s8.s16: %d\n", static_cast<signed char>(out));
    auto res2 = t2s8<short>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

/**
 *
 * @brief 无符号1位整数转有符号8位整数
 */
__global__ void test_cvt_sat_s8_u16() {
    unsigned short input = 300; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s8.u16 %0, %1;" : "=r"(out) : "h"(input));

    // 输出结果
    printf("Result of cvt.sat.s8.u16: %d\n", static_cast<signed char>(out));
    auto res2 = t2s8<unsigned short>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


/**
 * @brief 有符号32位整数转有符号8位整数
 */
__global__ void test_cvt_sat_s8_s32() {
    int input = -300; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s8.s32 %0, %1;" : "=r"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.s8.s32: %d\n", static_cast<signed char>(out));
    auto res2 = t2s8<int>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


/**
 * @brief 无符号32位整数转有符号8位整数
 */
__global__ void test_cvt_sat_s8_u32() {
    unsigned int input = 300; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s8.u32 %0, %1;" : "=r"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.s8.u32: %d\n", static_cast<signed char>(out));
    auto res2 = t2s8<unsigned int>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


/**
 *
 * @brief 单精度浮点数转有符号8位整数
 */
__global__ void test_cvt_rni_sat_s8_f32() {
    float input = -60.7f; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(out) : "f"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.s8.f32: %d\n", static_cast<signed char>(out));
    auto res2 = t2s8_rnd_float(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

/**
 *
 * @brief 双精度浮点数转有符号8位整数
 */
__global__ void test_cvt_rni_sat_s8_f64() {
    double input = -150.7; // 输入值超出 signed char 的范围 [-128, 127]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(out) : "d"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.s8.f64: %d\n", static_cast<signed char>(out));
    auto res2 = t2s8_rnd_double(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

///////////////////////////////////////// 无符号16位 /////////////////////////////////////////////////

__global__ void test_cvt_sat_u16_s8() {
    signed char input = -100; // 输入值为负数，超出 unsigned short 的范围 [0, 65535]
    unsigned short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u16.s8 %0, %1;" : "=h"(out) : "r"(static_cast<int>(input)));

    // 输出结果
    printf("Result of cvt.sat.u16.s8: %u\n", out);
    auto res2 = t2u16<signed char>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


__global__ void test_cvt_sat_u16_s16() {
    short input = -200; // 输入值为负数，超出 unsigned short 的范围 [0, 65535]
    unsigned short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u16.s16 %0, %1;" : "=h"(out) : "h"(input));

    // 输出结果
    printf("Result of cvt.sat.u16.s16: %u\n", out);
    auto res2 = t2u16<short>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


__global__ void test_cvt_sat_u16_s32() {
    int input = -50000; // 输入值为负数，超出 unsigned short 的范围 [0, 65535]
    unsigned short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u16.s32 %0, %1;" : "=h"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.u16.s32: %u\n", out);
    auto res2 = t2u16<int>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


__global__ void test_cvt_sat_u16_u32() {
    unsigned int input = 70000; // 输入值超出 unsigned short 的范围 [0, 65535]
    unsigned short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u16.u32 %0, %1;" : "=h"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.u16.u32: %u\n", out);
    auto res2 = t2u16<unsigned int>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


__global__ void test_cvt_rni_sat_u16_f32() {
    float input = -100.7f; // 输入值为负数，超出 unsigned short 的范围 [0, 65535]
    unsigned short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.u16.f32 %0, %1;" : "=h"(out) : "f"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.u16.f32: %u\n", out);
    auto res2 = t2u16_rnd_float(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}


__global__ void test_cvt_rni_sat_u16_f64() {
    double input = -100.7; // 输入值为负数，超出 unsigned short 的范围 [0, 65535]
    unsigned short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.u16.f64 %0, %1;" : "=h"(out) : "d"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.u16.f64: %u\n", out);
    auto res2 = t2u16_rnd_double(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

///////////////////////////////////////// 有符号16位 /////////////////////////////////////////////////

__global__ void test_cvt_sat_s16_u16() {
    unsigned short input = 40000; // 输入值超出 short 的范围 [-32768, 32767]
    short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s16.u16 %0, %1;" : "=h"(out) : "h"(input));

    // 输出结果
    printf("Result of cvt.sat.s16.u16: %d\n", out);
    auto res2 = t2u16<unsigned short>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_cvt_sat_s16_s32() {
    int input = -50000; // 输入值超出 short 的范围 [-32768, 32767]
    short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s16.s32 %0, %1;" : "=h"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.s16.s32: %d\n", out);
    auto res2 = t2u16<int>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_cvt_sat_s16_u32() {
    unsigned int input = 50000; // 输入值超出 short 的范围 [-32768, 32767]
    short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s16.u32 %0, %1;" : "=h"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.s16.u32: %d\n", out);
    auto res2 = t2u16<unsigned int>(input);
    printf("Result of device call: %u\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_cvt_rni_sat_s16_f32() {
    float input = -40000.7f; // 输入值超出 short 的范围 [-32768, 32767]
    short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.s16.f32 %0, %1;" : "=h"(out) : "f"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.s16.f32: %d\n", out);
    auto res2 = t2u16_rnd_float(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_cvt_rni_sat_s16_f64() {
    double input = -40000.7; // 输入值超出 short 的范围 [-32768, 32767]
    short out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.rni.sat.s16.f64 %0, %1;" : "=h"(out) : "d"(input));

    // 输出结果
    printf("Result of cvt.rni.sat.s16.f64: %d\n", out);
    auto res2 = t2u16_rnd_double(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

///////////////////////////////////////// 有符号32位 /////////////////////////////////////////////////

__global__ void test_cvt_sat_s32_u32() {
    unsigned int input = 3000000000; // 输入值超出 int 的范围 [-2147483648, 2147483647]
    int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.s32.u32 %0, %1;" : "=r"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.s32.u32: %d\n", out);
    auto res2 = u32_2_s32(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

///////////////////////////////////////// 无符号32位 /////////////////////////////////////////////////


__global__ void test_cvt_sat_u32_s8() {
    signed char input = -100; // 输入值为负数，超出 unsigned int 的范围 [0, 4294967295]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u32.s8 %0, %1;" : "=r"(out) : "r"(static_cast<int>(input)));

    // 输出结果
    printf("Result of cvt.sat.u32.s8: %u\n", out);
    // 输出结果
    printf("Result of cvt.sat.s32.u32: %d\n", out);
    auto res2 = t2u32<signed char>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_cvt_sat_u32_s16() {
    short input = -200; // 输入值为负数，超出 unsigned int 的范围 [0, 4294967295]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u32.s16 %0, %1;" : "=r"(out) : "h"(input));

    // 输出结果
    printf("Result of cvt.sat.u32.s16: %u\n", out);
    auto res2 = t2u32<short>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void test_cvt_sat_u32_s32() {
    int input = -1000; // 输入值为负数，超出 unsigned int 的范围 [0, 4294967295]
    unsigned int out;

    // 使用 PTX 汇编指令进行饱和转换
    asm("cvt.sat.u32.s32 %0, %1;" : "=r"(out) : "r"(input));

    // 输出结果
    printf("Result of cvt.sat.u32.s32: %u\n", out);
    auto res2 = t2u32<int>(input);
    printf("Result of device call: %d\n", res2);
    printf("Size of res2: %lu\n", sizeof(res2));
}

__global__ void float2int() {
    int r2i = __float2int_rd(120.5);
    printf("\n%d", r2i);
    r2i = __float2int_rd(121.4);
    printf("\n%d", r2i);
    r2i = __float2int_rd(122.4);
    printf("\n%d", r2i);
    r2i = __float2int_rd(123.4);
    printf("\n%d", r2i);
}

/**
 * @brief
 * @return
 */
int main() {

    float2int<<<1,1>>>();
    cudaDeviceSynchronize();

    // std::cout << "Call ASM: cvt.sat.u8.s8" << std::endl;
    // test_cvt_sat_u8_s8<<<1, 1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u8.s16" << std::endl;
    // test_cvt_sat_u8_s16<<<1, 1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u8.u16" << std::endl;
    // test_cvt_sat_u8_u16<<<1, 1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u8.s32" << std::endl;
    // test_cvt_sat_u8_s32<<<1, 1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u8.u32" << std::endl;
    // test_cvt_sat_u8_u32<<<1, 1>>>();
    // cudaDeviceSynchronize();

    // std::cout << "Call ASM: cvt.rni.sat.u8.f32" << std::endl;
    // test_saturate_cast<<<1, 1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.rni.sat.u8.f64" << std::endl;
    // test_cvt_rni_sat_u8_f64<<<1, 1>>>();
    // cudaDeviceSynchronize();

    // std::cout << "------------------------------s8------------------------------" << std::endl;
    //
    // std::cout << "Call ASM: cvt.sat.s8.u8" << std::endl;
    // test_cvt_sat_s8_u8<<<1, 1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Cal ASM: cvt.sat.s8.s16" << std::endl;
    // test_cvt_sat_s8_s16<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Cal ASM: cvt.sat.s8.u16" << std::endl;
    // test_cvt_sat_s8_u16<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Cal ASM: cvt.sat.s8.s32" << std::endl;
    // test_cvt_sat_s8_s32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Cal ASM: cvt.sat.s8.u32" << std::endl;
    // test_cvt_sat_s8_u32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Cal ASM: cvt.sat.s8.f32" << std::endl;
    // test_cvt_rni_sat_s8_f32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Cal ASM: cvt.sat.s8.f64" << std::endl;
    // test_cvt_rni_sat_s8_f64<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    //
    // std::cout << "------------------------------u16----------------------------" << std::endl;
    //
    // std::cout << "Call ASM: cvt.sat.u16.s8" << std::endl;
    // test_cvt_sat_u16_s8<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u16.s16" << std::endl;
    // test_cvt_sat_u16_s16<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u16.s32" << std::endl;
    // test_cvt_sat_u16_s32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u16.u32" << std::endl;
    // test_cvt_sat_u16_u32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.rni.sat.u16.f32" << std::endl;
    // test_cvt_rni_sat_u16_f32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.rni.sat.u16.f64" << std::endl;
    // test_cvt_rni_sat_u16_f64<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "------------------------------s16----------------------------" << std::endl;
    //
    // std::cout << "Call ASM: cvt.sat.s16.u16" << std::endl;
    // test_cvt_sat_s16_u16<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.s16.s32" << std::endl;
    // test_cvt_sat_s16_s32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.s16.u32" << std::endl;
    // test_cvt_sat_s16_u32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.rni.sat.s16.f32" << std::endl;
    // test_cvt_rni_sat_s16_f32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.rni.sat.s16.f64" << std::endl;
    // test_cvt_rni_sat_s16_f64<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    //
    // std::cout << "------------------------------s32----------------------------" << std::endl;
    //
    // std::cout << "Call ASM: cvt.sat.s32.u32" << std::endl;
    // test_cvt_sat_s32_u32<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "------------------------------u32----------------------------" << std::endl;
    //
    // std::cout << "Call ASM: cvt.sat.u32.s8" << std::endl;
    // test_cvt_sat_u32_s8<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u32.s16" << std::endl;
    // test_cvt_sat_u32_s16<<<1,1>>>();
    // cudaDeviceSynchronize();
    //
    // std::cout << "Call ASM: cvt.sat.u32.s32" << std::endl;
    // test_cvt_sat_u32_s32<<<1,1>>>();
    // cudaDeviceSynchronize();

    return 0;
}