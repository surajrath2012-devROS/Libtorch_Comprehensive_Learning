#include <torch/torch.h>
#include <iostream>

int main()
{
    //Create a 3x3 tensor filled with zeros
    torch::Tensor t = torch::zeros({3,3});
    std::cout << "A 3x3 zero tensor: " << std::endl;
    std::cout << t << std::endl;


    //Creating a tensor filled with random numbers
    torch::Tensor r = torch::rand({2,4});
    std::cout << "\nA 2x4 random tensor : " << std::endl;
    std::cout << r << std::endl;

    //Some Basic Math
    torch::Tensor a = torch::ones({3});
    torch::Tensor b = torch::ones({3});
    std::cout << "\na + b = " << (a + b) << std::endl;

    std::cout << "\nLibTorch is working! " << std::endl;

    return 0;
}