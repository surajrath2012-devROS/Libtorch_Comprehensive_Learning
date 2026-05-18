#include <torch/torch.h>
#include <iostream>
 
int main() {
    // Build a simple 3-layer MLP (Multi-Layer Perceptron)
    // Input: 784 features (like a flattened 28x28 MNIST image)
    // Output: 10 classes (digits 0-9)
    auto net = torch::nn::Sequential(
        torch::nn::Linear(784, 256),  // Layer 1: 784 -> 256
        torch::nn::ReLU(),             // Activation
        torch::nn::Dropout(0.3),       // Dropout 30%
        torch::nn::Linear(256, 128),  // Layer 2: 256 -> 128
        torch::nn::ReLU(),
        torch::nn::Dropout(0.2),
        torch::nn::Linear(128, 10)    // Layer 3: 128 -> 10
    );
 
    // Forward pass with dummy data
    auto input = torch::randn({32, 784}); // Batch of 32 images
    auto output = net->forward(input);
    std::cout << "Output shape: " << output.sizes() << std::endl; // {32, 10}
 
    // Count parameters
    long total_params = 0;
    for (auto& p : net->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    return 0;
}
 