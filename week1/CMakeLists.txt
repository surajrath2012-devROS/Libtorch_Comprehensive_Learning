cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_week1)

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)

# --- Day 1 App ---
add_executable(hello_tensor hello_tensor.cpp)
target_link_libraries(hello_tensor "${TORCH_LIBRARIES}")
set_property(TARGET hello_tensor PROPERTY CXX_STANDARD 17)

# --- Day 2 App ---
add_executable(tensor_ops tensor_operations.cpp)
target_link_libraries(tensor_ops "${TORCH_LIBRARIES}")
set_property(TARGET tensor_ops PROPERTY CXX_STANDARD 17)

# --- Day 3 App ---
add_executable(tensor_ops_2 tensor_operations_2.cpp)
target_link_libraries(tensor_ops_2 "${TORCH_LIBRARIES}")
set_property(TARGET tensor_ops_2 PROPERTY CXX_STANDARD 17)

# --- Day 4 App ---
add_executable(tensor_ops_3 tensor_operations_3.cpp)
target_link_libraries(tensor_ops_3 "${TORCH_LIBRARIES}")
set_property(TARGET tensor_ops_3 PROPERTY CXX_STANDARD 17)

# --- Day 5 App ---
add_executable(autograd autograd.cpp)
target_link_libraries(autograd "${TORCH_LIBRARIES}")
set_property(TARGET autograd PROPERTY CXX_STANDARD 17)