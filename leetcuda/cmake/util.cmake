
function(add_python_target)
    set(options)
    set(oneValueArgs TARGET_NAME SCRIPT_PATH WORKING_DIR)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(PARSE_ARGV 0 PYTHON_TARGET "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(NOT DEFINED PYTHON_TARGET_TARGET_NAME OR NOT DEFINED PYTHON_TARGET_SCRIPT_PATH)
        message(FATAL_ERROR "TARGET_NAME and SCRIPT_PATH are required arguments")
    endif()

    if(NOT DEFINED PYTHON_TARGET_WORKING_DIR)
        set(PYTHON_TARGET_WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    add_custom_target(${PYTHON_TARGET_TARGET_NAME}
            COMMAND ${Python3_EXECUTABLE} ${PYTHON_TARGET_SCRIPT_PATH} ${PYTHON_TARGET_ARGS}
            WORKING_DIRECTORY ${PYTHON_TARGET_WORKING_DIR}
            COMMENT "Running Python script: ${PYTHON_TARGET_SCRIPT_PATH}"
            VERBATIM
    )

endfunction()

## 调用函数创建目标
#add_python_target(
#        TARGET_NAME run_elementwise
#        SCRIPT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/leetcuda/kernels/elemenwise/elementwise.py
#        WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR}  # 可选
#        ARGS "--verbose" "param1" "param2"      # 可选参数
#)