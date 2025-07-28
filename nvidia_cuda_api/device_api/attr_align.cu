//
// Created by Gary27 on 2025/4/7.
//

#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>


template<typename U>
__device__ __forceinline__ void DeviceSqrtImplAllBranch(U u)
{
    printf("__fsqrt_rn: %.10f\n", __fsqrt_rn(u));

    printf("__dsqrt_rn: %.10f\n", __dsqrt_rn(u));

    printf("__fsqrt_rn: %.10f\n", static_cast<U>(__fsqrt_rn(static_cast<float>(u))));

    printf("__dsqrt_rn: %.10f\n", static_cast<U>(__dsqrt_rn(static_cast<double>(u))));

    printf("std::sqrt: %.10f\n", std::sqrt(u));
}



__global__ void sqrt_call() {
    float num = 29.41599083;
    DeviceSqrtImplAllBranch(num);

    printf("--------double gpu-------\n");
    double numd = 29.41599083;
    DeviceSqrtImplAllBranch(numd);


    printf("---------------------------brightness-----------------------------\n");
    uint8_t pixel = 66;
    // brightness, contrast, brightness_shift, contrast_center
    double4 args {16695252.238261191, 1.4708945844490164, -537512424.02722478, 128};
    auto b_pixel = args.x * (args.y * (pixel - args.w) + args.w) + args.z;
    printf("%.10f\n", b_pixel);
    printf("--------------------------tmp log GPU-----------------------------\n");

    printf("args.y :%.10f\n", args.y);


    double tmp1 = pixel - args.w;
    // double tmp2 = static_cast<long double>(args.y) * static_cast<long double>(tmp1);
    // double tmp2 = __dmul_rn(args.y, tmp1);
    double tmp2 = args.y * -62;
    // double tmp2 = args.y * pixel - t_argy * args.w;
    double tmp3 = tmp2 + args.w;
    double tmp4 = args.x * tmp3;  // a * (b + c)  lead to precision loss
    double tmp5 = tmp4 + args.z;

    printf("tmp1: %.15f\n", tmp1);
    printf("tmp2: %.15f\n", tmp2);
    printf("tmp3: %.15f\n", tmp3);
    printf("tmp4: %.15f\n", tmp4);
    printf("tmp5: %.15f\n", tmp5);

    b_pixel = args.x * args.y * (pixel - args.w) + args.x * args.w + args.z;
    printf("%.10f\n", b_pixel);
}


int main(){

    // float num = 29.41599083;
    // auto cpu_sqrt_res = std::sqrt(num);
    // std::cout << "cpu_sqrt_res: " << std::setprecision(10) << cpu_sqrt_res << std::endl;
    //
    // sqrt_call<<<1,1>>>();
    // cudaDeviceSynchronize();

    int dev = 0;

    int pitchAlign;
    cudaDeviceGetAttribute(&pitchAlign, cudaDevAttrTexturePitchAlignment, dev);
    std::cout << "CUDA texturePitchAlignment: " << pitchAlign << std::endl;


    int addrAlign;
    cudaDeviceGetAttribute(&addrAlign, cudaDevAttrTextureAlignment, dev);
    std::cout << "CUDA textureAlignment: " << addrAlign << std::endl;



    auto i64max = std::numeric_limits<int64_t>::max();

    int64_t i64_overflow = i64max + 2;

    if (i64_overflow < i64max) {
        std::cout << "i64_overflow < i64max" << "\n";
        std::cout << "i64_overflow: " << i64_overflow << "\n";
    }


    return 0;
}