//
// Created by Gary27 on 2024/2/7.
//
#include <iostream>
#include <iomanip>

int main(){

    auto float_A = (double*) malloc(10 * sizeof(double));
    for(int i = 0; i < 10; ++i){
        float_A[i] = (double)(0xFFFFFFFF & 4294967295);  // 4294967295
    }

    for (int i = 0;i < 10;++i) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "float [i]: " << float_A[i] << " ";
    }

    std::cout << "size of float: " << sizeof(float) << std::endl;
    std::cout << "0xFFFFFFFF & 10000000010110000000ull: " << (0xFFFFFFFF & 4294967295) << std::endl;
    return 0;
}