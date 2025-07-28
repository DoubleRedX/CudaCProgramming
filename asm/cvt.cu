//
// Created by Gary27 on 2025/6/24.
//

#include <cstdio>


using F32 = float;
using S32 = int;

__device__ __inline__ S32   f32_to_s32_sat (F32 a) {
    S32 v;
    asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(v) : "f"(a));          // __float2int_rn
    return v;

}


__global__ void device_call() {
    F32 a1 = 12.1;
    F32 a2 = 12.2;
    F32 a3 = 12.3;
    F32 a4 = 12.4;
    F32 a5 = 12.5;
    F32 a6 = 12.6;
    F32 a7 = 12.7;
    F32 a8 = 12.8;
    F32 a9 = 12.9;
    F32 a10 = 13.0;
    F32 a11 = 13.1;
    F32 a12 = 13.2;
    F32 a13 = 13.3;
    F32 a14 = 13.4;
    F32 a15 = 13.5;
    F32 a16 = 13.6;
    F32 a17 = 13.7;
    F32 a18 = 13.8;
    F32 a19 = 13.9;
    F32 a20 = 14.0;
    F32 a21 = 14.1;

    auto res1 = f32_to_s32_sat(a1);
    auto res2 = f32_to_s32_sat(a2);
    auto res3 = f32_to_s32_sat(a3);
    auto res4 = f32_to_s32_sat(a4);
    auto res5 = f32_to_s32_sat(a5);
    auto res6 = f32_to_s32_sat(a6);
    auto res7 = f32_to_s32_sat(a7);
    auto res8 = f32_to_s32_sat(a8);
    auto res9 = f32_to_s32_sat(a9);
    auto res10 = f32_to_s32_sat(a10);
    auto res11 = f32_to_s32_sat(a11);
    auto res12 = f32_to_s32_sat(a12);
    auto res13 = f32_to_s32_sat(a13);
    auto res14 = f32_to_s32_sat(a14);
    auto res15 = f32_to_s32_sat(a15);
    auto res16 = f32_to_s32_sat(a16);
    auto res17 = f32_to_s32_sat(a17);
    auto res18 = f32_to_s32_sat(a18);
    auto res19 = f32_to_s32_sat(a19);
    auto res20 = f32_to_s32_sat(a20);
    auto res21 = f32_to_s32_sat(a21);

    printf("res1 | ori: %f, cvt: %d\n", a1, res1);
    printf("res2 | ori: %f, cvt: %d\n", a2, res2);
    printf("res3 | ori: %f, cvt: %d\n", a3, res3);
    printf("res4 | ori: %f, cvt: %d\n", a4, res4);
    printf("res5 | ori: %f, cvt: %d\n", a5, res5);
    printf("res6 | ori: %f, cvt: %d\n", a6, res6);
    printf("res7 | ori: %f, cvt: %d\n", a7, res7);
    printf("res8 | ori: %f, cvt: %d\n", a8, res8);
    printf("res9 | ori: %f, cvt: %d\n", a9, res9);
    printf("res10 | ori: %f, cvt: %d\n", a10, res10);
    printf("res11 | ori: %f, cvt: %d\n", a11, res11);
    printf("res12 | ori: %f, cvt: %d\n", a12, res12);
    printf("res13 | ori: %f, cvt: %d\n", a13, res13);
    printf("res14 | ori: %f, cvt: %d\n", a14, res14);
    printf("res15 | ori: %f, cvt: %d\n", a15, res15);
    printf("res16 | ori: %f, cvt: %d\n", a16, res16);
    printf("res17 | ori: %f, cvt: %d\n", a17, res17);
    printf("res18 | ori: %f, cvt: %d\n", a18, res18);
    printf("res19 | ori: %f, cvt: %d\n", a19, res19);
    printf("res20 | ori: %f, cvt: %d\n", a20, res20);
    printf("res21 | ori: %f, cvt: %d\n", a21, res21);

}



int main() {
    device_call<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}