//
// Created by Gary27 on 2025/3/24.
//


#include "DropCast.h"

// \usr\local\cuda-11.8\targets\x86_64-linux\include\device_launch_parameters.h
// threadIdx 在cuda中是unit3
// blockIdx 在cuda中是uint3
// blockDim 在cuda中是dim3
// gridDim 在cuda中是dim3


__global__ void dc_test(){

    auto s = threadIdx;
    // auto num = detail::TypeTraits<>::components;

    //    if (HasEnoughComponents<__HIP_Coordinates<__HIP_ThreadIdx>, 2>){
    //
    //    }

    // int2 tc2 = StaticCast<int>(DropCast<2>(threadIdx));
    // printf("tc2: %d, %d", tc2.x, tc2.y);
}

int main(){

    dc_test<<<1,1>>>();
    cudaDeviceSynchronize();

    return 0;
}

