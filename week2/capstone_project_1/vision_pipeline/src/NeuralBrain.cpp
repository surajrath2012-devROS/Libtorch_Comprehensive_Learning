#include "NeuralBrain.h"
#include <iostream>

NeuralBrain::NeuralBrain() {
    std::cout << "[NeuralBrain] Initializing Deep Learning architecture..." << std::endl;
    
    // 784 -> 128 -> 64 -> 10
    fc1 = register_module("fc1", torch::nn::Linear(INPUT_FEATURES, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 64));
    fc3 = register_module("fc3", torch::nn::Linear(64, NUM_CLASSES));
}

torch::Tensor NeuralBrain::forward(torch::Tensor x) {
    x = x.view({x.size(0), INPUT_FEATURES}); // Flatten
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return torch::log_softmax(fc3->forward(x), /*dim=*/1);
}

int NeuralBrain::predict_digit(const torch::Tensor& image_frame) {
    // 1. Memory Safety: Disable the Autograd tracking engine completely
    torch::NoGradGuard no_grad;

    // 2. Add a batch dimension. The network expects [Batch, Features], not just [Features]
    // We use unsqueeze(0) to turn a [28, 28] image into a [1, 28, 28] batch
    torch::Tensor batched_image = image_frame.unsqueeze(0);

    // 3. Run the forward pass
    torch::Tensor output_probabilities = this->forward(batched_image);

    // 4. Find the index of the highest probability (argmax)
    // .item<int>() safely extracts the raw C++ integer from the LibTorch tensor
    int predicted_digit = output_probabilities.argmax(1).item<int>();

    return predicted_digit;
}