#include <torch/torch.h>
 
// ── Define the network architecture ──────────────────────────
struct MyCNNImpl : torch::nn::Module {
    // Declare layers as members
    torch::nn::Conv2d conv1, conv2;
    torch::nn::Linear fc1, fc2;
    torch::nn::BatchNorm2d bn1, bn2;
    torch::nn::Dropout dropout;
 
    // Constructor: define and register layers
    MyCNNImpl(int num_classes = 10)
        : conv1(torch::nn::Conv2dOptions(1, 32, 3).padding(1)),   // 1 ch in, 32 out, 3x3 kernel
          conv2(torch::nn::Conv2dOptions(32, 64, 3).padding(1)), // 32 in, 64 out
          bn1(32), bn2(64),
          fc1(64 * 7 * 7, 128),  // After 2 max-pool ops on 28x28: 7x7
          fc2(128, num_classes),
          dropout(0.5) {
        // CRITICAL: Register all submodules so they appear in parameters()
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("bn1",   bn1);
        register_module("bn2",   bn2);
        register_module("fc1",   fc1);
        register_module("fc2",   fc2);
        register_module("dropout", dropout);
    }
 
    // Forward pass: define how data flows through the network
    torch::Tensor forward(torch::Tensor x) {
        // Block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 2);  // Halve spatial dimensions
 
        // Block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        x = conv2->forward(x);
        x = bn2->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 2);
 
        // Flatten: {batch, 64, 7, 7} -> {batch, 64*7*7}
        x = x.view({x.size(0), -1});
 
        // Fully connected layers
        x = torch::relu(fc1->forward(x));
        x = dropout->forward(x);
        x = fc2->forward(x);  // No activation here — loss fn handles it
        return x;
    }
};
TORCH_MODULE(MyCNN);  // Creates MyCNN as a shared_ptr wrapper (standard practice)
 
int main() {
    std::cout << "--- Initializing CNN Training ---" << std::endl;
    
    MyCNN model(10);             
    model->train(); // Enables Dropout and BatchNorm tracking

    // 1. Setup the Mechanic (Optimizer) and the Grader (Loss Function)
    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(0.001));
    auto criterion = torch::nn::CrossEntropyLoss();

    // 2. Create a static simulated dataset to memorize
    // 32 images (1 channel, 28x28 pixels) and 32 target labels (0 through 9)
    auto train_images = torch::randn({32, 1, 28, 28});
    auto train_labels = torch::randint(0, 10, {32}, torch::TensorOptions().dtype(torch::kLong));

    // 3. The 5-Step Training Cycle
    for (int epoch = 1; epoch <= 50; ++epoch) {
        
        // STEP 1: Clear the board (wipe old gradients)
        optimizer.zero_grad();

        // STEP 2: The Test (Forward Pass)
        auto outputs = model->forward(train_images);

        // STEP 3: The Grade (Compute how wrong the AI is)
        auto loss = criterion(outputs, train_labels);

        // STEP 4: The Review (Autograd calculates the required weight changes)
        loss.backward();

        // STEP 5: The Adjustment (The Optimizer turns the mathematical knobs)
        optimizer.step();

        // Log the progress
        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << "[Training] Epoch: " << epoch 
                      << " | Loss Error: " << loss.item<float>() << std::endl;
        }
    }

    std::cout << "--- Training Complete ---" << std::endl;
    return 0;
}