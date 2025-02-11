//
// Created by Gary27 on 2025/1/16.
//

#include <cstdio>

__device__ __forceinline__ float cal_rcp_fp32(int64_t _a) {
    int64_t a = abs(_a);

    float rcp;
    asm("rcp.approx.f32 %0, %1;" : "=f"(rcp) : "f"(__int2float_rn(a)));
    return rcp;
}

__device__ __forceinline__ double cal_rcp_fp64(double a) {
    double rcp;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(rcp) : "d"(a));
    return rcp;
}


__global__ void test_rcp(){
    int64_t a = 22222222;
    auto r = cal_rcp_fp32(a);
    printf("fp32 | a_rcp: %.20f\n", r);


    double ad = 222222222;
    auto rcp64 = cal_rcp_fp64(ad);
    printf("fp64 | a_rcp: %.20f\n", rcp64);
}


int main(){

    test_rcp<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}