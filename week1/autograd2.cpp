/* Let us take up a scenario where we have a robot and it is looking at a 
bunch of messy data points on a graph. We as engineers, know that, this 
follows a hidden rule : y = 2x + 1. But the computer does not know this! 

Our goal is to give the compiler a random guess of y = w*x + b, show it the data, 
and let it use Gradient Descent(Autograd). It should slowly adjust until it finds out
that w should be 2.0 and b should be 1.0 */

#include <torch/torch.h>
#include <iostream>

int main()
{
// ── STEP 1: Create fake training data ──────────────────
// True relationship: y = 2*x + 1  (we are trying to LEARN this)

auto x = torch::linspace(-1, 1, 100).unsqueeze(1);            // Creates 100 evenly spaced numbers between -1 and 1 and unsqueeze(1) bends it into a 2D matrix of shape [100, 1]
auto y_true = 2.0 *x + 1.0 + 0.1 * torch::randn_like(x);      // add noise


// ── STEP 2: Initialize learnable parameters ────────────
// We are guessing: y = w*x + b
// Start with random w and b

auto w = torch::randn({1, 1}, torch::requires_grad(true));
auto b = torch::zeros({1}, torch::requires_grad(true));
float lr = 0.01;  // Learning rate

// ── STEP 3: Training loop ───────────────────────────────

for (int epoch = 0; epoch < 200; epoch++) {
        // Forward pass: compute prediction
        auto y_pred = x.mm(w) + b;  // matrix multiply + bias

        // Compute loss (Mean Squared Error)
        auto loss = (y_pred - y_true).pow(2).mean();
 
        // Backward pass: compute gradients
        loss.backward();
        
        // Update parameters (gradient descent, NO autograd here)
        {
            torch::NoGradGuard no_grad;          // MUST disable gradient tracking during update
            w -= lr * w.grad();
            b -= lr * b.grad();
        }
 
        // Reset gradients for next iteration (VERY IMPORTANT!)
        w.grad().zero_();
        b.grad().zero_();
 
        if (epoch % 50 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Loss: " << loss.item<float>()
                      << " | w: " << w.item<float>()
                      << " | b: " << b.item<float>()
                      << std::endl;
        }
    }
    // After training: w ≈ 2.0, b ≈ 1.0  — it LEARNED the relationship!
    return 0;
}