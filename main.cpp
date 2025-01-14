//
// Created by Gary27 on 2025/1/9.
//

#include <iostream>



int main(){


    for(int i=1;i<=32;i*=2){
        std::cout << "i: " << i << std::endl;
    }
    int a = 1;
    std::cout << "a: " << a << std::endl;
    a >>= 2;
    std::cout << "a: " << a << std::endl;

    return 0;
}