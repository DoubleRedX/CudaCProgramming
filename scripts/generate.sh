#!/usr/bin/bash

if [ -d "build" ]; then
  echo "build目录已存在，正在删除..."
  rm -rf build
  echo "build目录删除成功..."
fi

mkdir build
echo "build目录已重新创建..."

echo "开始生成构建树..."
cmake --version

/opt/conda/bin/cmake -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_TOOLCHAIN_FILE=/data3/cll/codes/cprojects/book_cuda_c_programming/toolchain/weitu11.cmake \
-G "Unix Makefiles" \
-S /data3/cll/codes/cprojects/book_cuda_c_programming \
-B /data3/cll/codes/cprojects/book_cuda_c_programming/build

