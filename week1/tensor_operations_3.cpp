#include <torch/torch.h>
#include <iostream>


int main()
{
// Matrix Multiplication Operations in LibTorch
auto mat1 = torch::randn({3,4});        // Matrix 1
std::cout << mat1 << std::endl;
auto mat2 = torch::randn({4,3});        // Matrix 2
std::cout << "\n" << mat2 << std::endl;
std::cout <<"\n ---------- This is one type of Matrix Multiplication ------" << std::endl;

auto matmul = torch::mm(mat1, mat2);    // One way of conducting Matrix Multiplication
std::cout << matmul << std::endl;

std::cout <<"\n ---------- And this one below is the second type of Matrix Multiplication ------" << std::endl;
auto t_matmul = mat1.matmul(mat2);      // Another way of conducting a matrix multiplication 
std::cout << t_matmul << std::endl;

// Reduction Operation on Matrices
auto t = torch::randn({4,4});
auto total = t.sum();                   // Sum of ALL elements
std::cout << "\nSum of all element of the newly created random normal tensor : " << total << std::endl;
auto row_sum = t.sum(1);                // Sum along dimension 1 (per row)
std::cout << "\nSum along dimension 1 : " << row_sum << std::endl;
auto col_mean = t.mean(0);              // Mean along dimension 0
std::cout << "\nMean along dimension 0 : " << col_mean << std::endl;
auto max_val = t.max();                 // Maximum value
std::cout << "\nThe maximum value in the matrix is : " << t.max() << std::endl;
auto min_val = t.min();                 // Minimum value
std::cout << "\nThe minimum value in the matrix is : " << min_val << std::endl;


// We can manipulate matrices in certain ways such that dimensions can be added or removed
auto v = torch::ones({3});
auto row_vec = v.unsqueeze(0);           // Shape: {1, 3}
std::cout << "\nDisplaying the row vector : " << row_vec << std::endl;

auto col_vec = v.unsqueeze(1);           // Shape: {3, 1}
std::cout << "\nDisplaying column vector : " << col_vec << std::endl;

auto back = col_vec.squeeze(1);           // Back to {3}
std::cout << "\nPutting back column vector to a singular design : " << back << std::endl;

    return 0;
}