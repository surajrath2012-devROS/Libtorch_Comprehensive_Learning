cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
PROJECT(libtorch_day1)

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(app main.cpp)
target_link_libraries(app "${TORCH_LIBRARIES}")
set_property(TARGET app PROPERTY CXX_STANDARD 17)