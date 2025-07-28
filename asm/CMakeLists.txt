#
#find_package(CUDAToolkit REQUIRED)
#
#if(CUDAToolkit_FOUND)
#    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
#    message(STATUS "CUDAToolkit_LIBRARIES: ${CUDAToolkit_LIBRARIES}")
#endif ()
#
#include_directories(${CUDAToolkit_INCLUDE_DIRS})




add_executable(bfind bfind.cu)

add_executable(bfi bfi.cu)
add_executable(bfe bfe.cu)
add_executable(rcp rcp.cu)
add_executable(mul mul.cu)
add_executable(sqrt sqrt.cu)

add_executable(shl shl.cu)
add_executable(sub sub.cu)
add_executable(shr shr.cu)

add_executable(cvt_rni_sat_u8_f32 cvt_rni_sat_u8_f32.cu)

add_subdirectory(compose)

add_executable(data_type data_type.cu)

add_executable(declval_tt declval_tt.cu)


add_executable(cvt cvt.cu)