#pragma once
#include <torch/torch.h>
#include <string>

// Creating an Abstract Base Class (The Interface)
class BaseSensor {
public: 
    // Virtual destructor ensures proper cleanup of derived classes
    virtual ~BaseSensor() = default;

    // Pure Virtual function called capture_frame, this will force all child classes to implement the same
    virtual torch::Tensor capture_frame() = 0;

};

// Concrete Class (The Implementation)
class MnistCamera : public BaseSensor {
public:
    // Constructor and Destructor
    MnistCamera(const std::string& image_path);
    ~MnistCamera() override;

    // Fulfilling the abstract contract
    torch::Tensor capture_frame() override; 


private:
    std::string file_path;
    int current_index;
    torch::Tensor preloaded_images;              // The buffer holding our camera feed

    // Internal helper function to parse raw binary
    void load_mnist_binary(const std::string& path);

};