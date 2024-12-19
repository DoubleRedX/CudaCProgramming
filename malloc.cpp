//
// Created by Gary27 on 2024/2/7.
//
#include <iostream>
#include <iomanip>

int main() {

    int A[3][4]{
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12

    };

    printf("A[1][0] : %d", A[1][0]);
    printf("A[1][0] : %d", A[0][1]);

//    auto float_A = (double *) malloc(10 * sizeof(double));
//    for (int i = 0; i < 10; ++i) {
//        float_A[i] = (double) (0xFFFFFFFF & 4294967295);  // 4294967295
//    }
//
//    for (int i = 0; i < 10; ++i) {
//        std::cout << std::fixed << std::setprecision(2);
//        std::cout << "float [i]: " << float_A[i] << " ";
//    }
//
//    std::cout << "size of float: " << sizeof(float) << std::endl;
//    std::cout << "0xFFFFFFFF & 10000000010110000000ull: " << (0xFFFFFFFF & 4294967295) << std::endl;
    return 0;
}