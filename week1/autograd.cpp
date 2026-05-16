#include <torch/torch.h>
#include <iostream>


int main() {

// By Default tensors do not track gradient
auto x = torch::tensor(3.0);
std::cout << x.requires_grad() << std::endl;                   // 0 (false)

// Enable gradient tracking
auto w = torch::tensor(2.0, torch::requires_grad(true));
// OR :
auto w2 = torch::rand({3,3}).requires_grad_(true);

// Now let's do some math and compute gradients
// y = w * x^2 + 3
auto x_val = torch::tensor(3.0);                               // x = 3, no gradient needed
auto w_val = torch::tensor(2.0, torch::requires_grad(true));   // w = 2                   

auto y = w_val * x_val.pow(2) + 3;                             // y = 2*(3^2) + 3 = 21
std::cout << "y = " << y.item<float>() << std::endl;           // 21


// Compute gradients (backward pass)
y.backward();

// dy/dw = x^2 = 9
std::cout << "dy/dw = " << w_val.grad() << std::endl;           // 9
// This means to reduce y, decrease w by 9 * learning_rate

    return 0;
}