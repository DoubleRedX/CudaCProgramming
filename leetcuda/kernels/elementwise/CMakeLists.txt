add_python_target(
        TARGET_NAME run_elementwise
        SCRIPT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/leetcuda/kernels/elementwise/elementwise.py
        WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}  # 可选
)


add_executable(elementwise elementwise.cu)