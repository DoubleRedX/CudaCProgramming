//
// Created by Gary27 on 2025/4/2.
//

#include <iostream>
#include <list>
#include <vector>

// 两个不同类型的容器
std::list<int> lst = {1, 2, 3};
std::vector<double> vec = {4.5, 5.5, 6.5};

int main() {
    // GCC 11 允许 auto 推导不同的迭代器类型
    // auto it_lst = lst.begin();
    // auto it_vec = vec.begin();

    for (auto it_lst = lst.begin(), it_vec = vec.begin(); it_lst != lst.end() && it_vec != vec.end(); ++it_lst, ++it_vec) {
        std::cout << "List: " << *it_lst << ", Vector: " << *it_vec << std::endl;
    }

    return 0;
}