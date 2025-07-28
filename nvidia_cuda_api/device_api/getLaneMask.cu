//
// Created by Gary27 on 2025/6/23.
//

#include <iostream>
#include <cuda_runtime.h>


// __device__ __inline__ U32   getLaneMaskLt           (void)                  { U32 r; asm("mov.u32 %0, %lanemask_lt;" : "=r"(r)); return r; }

using U32 = uint32_t;

__device__ __inline__ U32 getLaneMaskLt() {
    U32 r;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(r));
    return r;
}

__global__ void getLaneMaskLtKernel() {
    auto lane_id = threadIdx.x % warpSize;
    auto laneMask = getLaneMaskLt();

    printf("Thread %2d: laneMaskLt = 0x%08x\n", lane_id, laneMask);
}

__global__ void debug() {
    unsigned int tidx = threadIdx.x;
    unsigned int tidy = threadIdx.y;
    unsigned int blkx = blockDim.x;

    int num = 0;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        num = 1;
    }
    U32 myIdx = __popc(__ballot_sync(~0u, num & 1) & getLaneMaskLt());

    // if (threadIdx.x == 31 && threadIdx.y == 1) {
    printf("thidx = (%d, %d), myIdx = %d, blockDim.x = %d\n", tidx, tidy , myIdx, blkx);
}

using S32 = int32_t;



__device__ __inline__ U32   add_sub					(U32 a, U32 b, U32 c)	{ U32 v; asm("vsub.u32.u32.u32.add %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(c), "r"(b)); return v; }

__global__ void debug3() {
    int a = 10;
    int b = 20;
    int c = 3;
    auto res = add_sub(a, b, c);
    printf("res = %d\n", res);
}

__device__ __inline__ U32 prmt(U32 a, U32 b, U32 c)
{
    U32 v;
    asm("prmt.b32 %0, %1, %2, %3;"
        : "=r"(v)
        : "r"(a), "r"(b), "r"(c));
    return v;
}

__global__ void test_prmt()
{
    U32 a = 0x11223344;
    U32 b = 0xAABBCCDD;
    U32 c = 0x01234567;

    U32 result = prmt(a, b, c);
    printf("prmt result = 0x%08X\n", result);
}


__device__ __inline__ S32   sub_s16lo_s16lo (S32 a, S32 b) {
    S32 v;
    asm("vsub.s32.s32.s32 %0, %1.h0, %2.h0;" : "=r"(v) : "r"(a), "r"(b));
    return v;
}

__device__ __inline__ S32 sub_s16lo_s16lo_SIMU (S32 a, S32 b) {
    S32 v = (a & 0xFFFF) - (b & 0xFFFF);
    return v;
}


__global__ void debug4() {
    unsigned int a = 4187683226;
    unsigned int b = 0;
    int res = sub_s16lo_s16lo(a, b);
    printf("sub_s16lo_s16lo(%d, %d) = %d\n", a, b, res);

    int res2 = sub_s16lo_s16lo_SIMU(a, b);
    printf("sub_s16lo_s16lo_SIMU(%d, %d) = %d\n", a, b, res2);

}

__global__ void debug5(){

    unsigned int a = 1;
    int res = __clz(a);
    printf("ddddd: %d\n", res);

}

// __device__ __inline__ U32   getLaneMaskLe () {
//     U32 r;
//     asm("mov.u32 %0, %lanemask_le;" : "=r"(r));
//     return r;
// }

__device__ __inline__ U32   getLaneMaskLe ( ) {
    U32 r;
    asm("mov.u32 %0, %lanemask_le;" : "=r"(r));
    return r;
}


__global__ void getLaneMaskLeKernel() {
    auto mask = getLaneMaskLe();
    if (threadIdx.x == 31 && threadIdx.y == 7) {
        printf("Thread %2d: getLaneMaskLe = 0x%08x\n", threadIdx.x, mask);

    }
}

__global__ void printThreadIdx() {

    printf("threadIdx.x = %d\n", threadIdx.x);

}

__device__ __inline__ S32 slct(S32 a, S32 b, S32 c)
{
    S32 v;
    asm("slct.s32.s32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c));
    return v;
}

__global__ void slct_test() {

    int a = 10;
    int b = 20;
    int c = 3;
    auto res = slct(a, b, c);
    printf("factor = %d, res = %d\n", c, res);
    int d = -3;
    res = slct(a, b, d);
    printf("factor = %d, res = %d\n", d, res);
    int e = 0;
    res = slct(a, b, e);
    printf("factor = %d, res = %d\n", e, res);

}


int main(){
    // debug<<<dim3(8,8,1), dim3(8, 8, 1)>>>();
    // getLaneMaskLeKernel<<<dim3(1, 1, 1), dim3(32, 8, 1)>>>();
    // printThreadIdx<<<dim3(2, 1, 1), dim3(32, 2, 1)>>>();
    slct_test<<<1,1>>>();
    cudaDeviceSynchronize();
    // std::cout << "1<<13: " << (1 << 13) - 1 << std::endl;
    return 0;
}