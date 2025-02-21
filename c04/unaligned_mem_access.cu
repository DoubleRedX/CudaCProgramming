//
// Created by Gary27 on 2025/2/21.
//

#include <cmdline.h>

__global__ void read_offset(float* A, float* B,  float* C, int n, int offset){

    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto k = i + offset;

}


int main(){

    cmdline::parser parser;
    parser.add<int>("offset", 'o', "offset address", false, 0);

    auto offset = parser.get<int>("offset");




    return 0;
}