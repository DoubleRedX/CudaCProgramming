add_executable(__active_mask active_mask.cu)
add_executable(shfl_examples shfl_examples.cu)

add_executable(shfl_down shfl_down.cu)


add_executable(shfl_xor shfl_xor.cu)
add_executable(shfl_up_sync_example shfl_up_sync_example.cu)
add_executable(shfl_down_sync_example shfl_down_sync_example.cu)
add_executable(shfl_xor_sync_example shfl_xor_sync_example.cu)

add_executable(cuda_solver cuda_solver.cu)
target_link_libraries(cuda_solver PUBLIC CUDA::cusolver)

add_executable(Stream Stream.cu)

add_executable(attr_align attr_align.cu)

add_executable(default_malloc default_malloc.cu)

add_executable(getLaneMask getLaneMask.cu)