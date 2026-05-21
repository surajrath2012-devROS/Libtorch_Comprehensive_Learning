#include <torch/torch.h>
#include <iostream>

struct DeepArchitectureImpl : torch::nn::Module {
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Dropout dropout;
    torch::nn::Linear fc1;

    DeepArchitectureImpl() 
        // 1 Channel In (Grayscale), 16 Channels Out, 3x3 Scanner Window
        : conv1(torch::nn::Conv2dOptions(1, 16, 3).padding(1)), 
          // Stabilizer expecting 16 channels
          bn1(16),
          // Randomly disable 50% of neurons
          dropout(0.5),
          // 16 channels * 14 height * 14 width (after max pooling down from 28x28)
          fc1(16 * 14 * 14, 10) {
        
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("dropout", dropout);
        register_module("fc1", fc1);
    }

    torch::Tensor forward(torch::Tensor x) {
        std::cout << "[Stage 0] Raw Input Image:    " << x.sizes() << "\n";

        // 1. Spatial Processing
        x = conv1->forward(x);
        std::cout << "[Stage 1] After Conv2d:       " << x.sizes() << " (Extracting features)\n";

        // 2. Stabilization
        x = bn1->forward(x);
        std::cout << "[Stage 2] After BatchNorm:    " << x.sizes() << " (Stabilizing math)\n";

        // 3. Non-Linearity
        x = torch::relu(x);
        std::cout << "[Stage 3] After ReLU:         " << x.sizes() << " (Filtering negative values)\n";

        // 4. Spatial Reduction
        x = torch::max_pool2d(x, 2);
        std::cout << "[Stage 4] After MaxPool:      " << x.sizes() << " (Shrinking image size by half)\n";

        // 5. Flattening for the Linear layer
        x = x.view({x.size(0), -1});
        std::cout << "[Stage 5] After Flattening:   " << x.sizes() << " (Converting 2D to 1D)\n";

        // 6. Regularization
        x = dropout->forward(x);
        std::cout << "[Stage 6] After Dropout:      " << x.sizes() << " (Randomly turning neurons off)\n";

        // 7. Final Classification
        x = fc1->forward(x);
        std::cout << "[Stage 7] After Linear:       " << x.sizes() << " (Mapping to 10 classes)\n";

        // 8. Probability Distribution
        x = torch::softmax(x, 1);
        std::cout << "[Stage 8] After Softmax:      " << x.sizes() << " (Converting to percentages)\n";

        return x;
    }
};
TORCH_MODULE(DeepArchitecture);

int main() {
    std::cout << "--- Initializing Architecture Flow Test ---\n\n";
    
    DeepArchitecture model;
    
    // Simulate a batch of 4 grayscale images, 28x28 pixels each
    auto dummy_batch = torch::randn({4, 1, 28, 28}); 
    
    // Pass the data through the network
    model->forward(dummy_batch);
    
    std::cout << "\n--- Pipeline Test Complete ---\n";
    return 0;
}