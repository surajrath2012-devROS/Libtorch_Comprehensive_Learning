cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(libtorch_week2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)

add_executable(custom_cnn custom_module_nn.cpp)
target_link_libraries(custom_cnn "${TORCH_LIBRARIES}")
set_property(TARGET custom_cnn PROPERTY CXX_STANDARD 17)
