#include "Sensor.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

// Utility function to swap endianness (MNIST files are high-endian, Intel CPUs are low-endian)
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

MnistCamera::MnistCamera(const std::string& image_path) : file_path(image_path), current_index(0) {
    std::cout << "[MnistCamera] Initializing sensor connection..." << std::endl;
    load_mnist_binary(file_path);
}

MnistCamera::~MnistCamera() {
    std::cout << "[MnistCamera] Sensor connection closed." << std::endl;
}

void MnistCamera::load_mnist_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open camera data stream: " + path);
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    // Read the 16-byte header
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    // To save RAM for this simulation, we will load a buffer of 100 images
    uint32_t images_to_load = std::min(num_images, (uint32_t)100);
    int image_size = num_rows * num_cols;

    // Allocate an empty LibTorch tensor for raw bytes
    auto options = torch::TensorOptions().dtype(torch::kByte);
    torch::Tensor buffer = torch::empty({images_to_load, num_rows, num_cols}, options);

    // Read the binary data directly into the LibTorch tensor's memory pointer
    file.read(reinterpret_cast<char*>(buffer.data_ptr()), images_to_load * image_size);

    // Convert to float, normalize to 0.0 to 1.0, and save to our class variable
    preloaded_images = buffer.to(torch::kFloat32).div(255.0);
    std::cout << "[MnistCamera] Successfully loaded " << images_to_load << " frames into buffer." << std::endl;
}

torch::Tensor MnistCamera::capture_frame() {
    if (current_index >= preloaded_images.size(0)) {
        current_index = 0; // Reset index to simulate a continuous video feed
    }
    return preloaded_images[current_index++];
}