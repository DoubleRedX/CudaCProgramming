#!/usr/bin/bash

# 检查是否传递参数
if [ $# -eq 0 ]; then
    TARGET="all"  # 默认目标
else
    TARGET="$1"   # 用户提供的目标
fi

/opt/conda/bin/cmake --build /data3/cll/codes/cprojects/book_cuda_c_programming/build --target "$TARGET" -- -j 16


