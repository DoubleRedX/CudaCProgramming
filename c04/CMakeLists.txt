add_executable(c04_static_global_mem static_global_mem.cu)
add_executable(unified_mem unified_mem.cu)
add_executable(cudaMalloc cudaMalloc.cu)
add_executable(cudaHostMalloc cudaHostMalloc.cu)

add_executable(mem_accesse_mode mem_access_mode.cu)
add_executable(unaligned_mem_access unaligned_mem_access.cu)
add_executable(read_only_mem read_only_cache.cu)
add_executable(unaligned_mem_write unaligned_mem_write.cu)

add_executable(AoS AoS.cu)
add_executable(SoA SoA.cpp)

add_executable(bandwidth bandwidth.cu)
add_executable(matrix_transpose matrix_transpose.cu)

