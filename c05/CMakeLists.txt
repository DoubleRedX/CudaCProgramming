add_executable(block_expand block_expand.cu)
target_compile_options(block_expand PUBLIC -g -G -O0)


add_executable(transpose transpose.cu)
