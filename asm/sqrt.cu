//
// Created by Gary27 on 2025/1/16.
//

#include <cstdio>

__device__ __forceinline__ double cal_sqrt_fp64(double a) {

    double rcp;
    asm("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(rcp) : "d"(a));
    return rcp;
}



__global__ void test_sqrt(){

    double ad = 100.;
    auto sqrt64 = cal_sqrt_fp64(ad);
    printf("fp64 | a_sqrt: %.20f\n", sqrt64);
}


int main(){

    test_sqrt<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}