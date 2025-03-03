//
// Created by Gary27 on 2025/3/3.
//

#include <iostream>

constexpr int N = 1 << 22;

struct s{

    float a[N];
    float b[N];

};


int main(){

    auto data_bytes = sizeof(s);

    std::cout << "Data bytes: " << data_bytes / 1024 /1024 << " MiB" << "\n";

    return 0;
}