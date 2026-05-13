// Understanding LibTorch Tensor Operations

#include <torch/torch.h>
using namespace torch::indexing;


int main()
{
    // Building a tensor of zeros
    torch::Tensor t = torch::zeros({3,4});
    std::cout << t << std::endl;


    // Building a tensor from an existing float
    float arr[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto from_arr = torch::from_blob(arr, {2,3});
    std::cout << from_arr << std::endl;

    // ---- Specifying and Working with Different Data Types ----
    auto float64 = torch::zeros({3}, torch::kFloat64);
    auto int32 = torch::zeros({3}, torch::kInt32);
    auto bool_t = torch::zeros({3}, torch::kBool);


    std::cout << float64 << std::endl;
    std::cout << int32 << std::endl;
    std::cout << bool_t << std::endl;

    /* Tensor Properties and Understanding Tensors further -->
    If let's say we have a tensor called t and we store random normal 
    numbers, what all operations can we do?*/

    torch::Tensor t_3d = torch::randn({4, 3, 2});

    // Shape and size 
    std::cout << t_3d.sizes() << std::endl;   // [4,3,2]
    std::cout << t_3d.size(0) << std::endl;   // 4
    std::cout << t_3d.size(1) << std::endl;   // 3
    std::cout << t_3d.numel() << std::endl;   // 24 (total elements : 4*3*2)
    std::cout << t_3d.dim() << std::endl;     // 3

    auto images = torch::rand({8, 3, 64, 64});

    // Slice: [Image Index, Channel Index, Height, Width]
    auto sliced_channel = images.index({0, 2, Slice(), Slice()});
    std::cout << "\nSliced Channel Shape: " << sliced_channel.sizes() << std::endl;

    return 0;
}