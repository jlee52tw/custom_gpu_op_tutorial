cmake_minimum_required(VERSION 3.10)
project(custom_gpu_op_tutorial_cpp)

find_package(OpenVINO REQUIRED)

add_executable(test_custom_op_cpp main.cpp)
target_link_libraries(test_custom_op_cpp PRIVATE openvino::runtime)

# Copy necessary files to build dir
file(COPY custom_add_mul.xml custom_add_mul.cl model.xml model.bin DESTINATION ${CMAKE_BINARY_DIR})
