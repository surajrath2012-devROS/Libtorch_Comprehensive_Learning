#pragma once
#include <torch/torch.h>

/**
 * @class NeuralBrain
 * @brief The AI inference engine. Inherits from LibTorch's nn::Module.
 */
class NeuralBrain : public torch::nn::Module {
public:
    // static constexpr: Modern C++17 way to define constants that are evaluated at compile-time
    static constexpr int IMAGE_WIDTH = 28;
    static constexpr int IMAGE_HEIGHT = 28;
    static constexpr int INPUT_FEATURES = IMAGE_WIDTH * IMAGE_HEIGHT;
    static constexpr int NUM_CLASSES = 10;
    
    /**
     * @brief Constructor initializes the network architecture.
     */
    NeuralBrain();

    /**
     * @brief The raw mathematical forward pass.
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief A safe, production-ready inference wrapper.
     * Takes an image, disables gradients, runs the network, and returns the digit.
     */
    int predict_digit(const torch::Tensor& image_frame);

private:
    // Network layers
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};