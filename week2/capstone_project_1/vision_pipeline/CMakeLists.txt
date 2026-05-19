cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(vision_pipeline)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED) # Find the OS threading library

set(CMAKE_CXX_STANDARD 17)
include_directories(include)

add_executable(pipeline src/main.cpp src/Sensor.cpp src/NeuralBrain.cpp)

# Link LibTorch AND the OS Threading library
target_link_libraries(pipeline "${TORCH_LIBRARIES}" Threads::Threads)
set_property(TARGET pipeline PROPERTY CXX_STANDARD 17)