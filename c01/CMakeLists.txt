include_directories(include)

file(GLOB AUX_SOURCES "./*.cpp")

add_library(
        c01_aux
        SHARED
        ${AUX_SOURCES}
)

set_target_properties(c01_aux PROPERTIES CXX_STANDARD 17)


link_libraries(c01_aux)

add_executable(c01_cuHelloWorld cuHelloWorld.cu)
add_executable(c01_vecAdd c01_main_pro_with_device_pro.cu)
add_executable(c01_kernel_timer_chrono c01_kernel_timer_chrono.cu)
add_executable(c01_kernel_timer_cudaEvent c01_kernel_timer_cudaEvent.cu)
add_executable(c01_kernel_organize_thread c01_organize_thread.cu)
add_executable(c01_matrix_sum_2dgrid_2dblock c01_matrix_sum_2dgrid_2dblock.cu)
