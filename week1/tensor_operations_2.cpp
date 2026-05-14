#include <torch/torch.h>
#include <iostream>

using namespace torch::indexing;

int main()
{
/* Working on further Tensor Operations ----> Obtaining Data Type, Obtaining Device (CPU or CUDA), If CUDA is available.
Further operations include indexing, slicing and more math operations... */

torch::Tensor num_x = torch::randn({4, 3, 2});

// Data Type
std::cout << num_x.dtype() << std::endl;       // returns the data type

// Device (CPU or CUDA)
std::cout << num_x.device() << std::endl;      // In this case it returns cpu

// Check if CUDA is available
if (torch::cuda::is_available()) {

    auto t_gpu = num_x.to(torch::kCUDA);        // Moves to CUDA
    std::cout << t_gpu.device() << std::endl;   // Outputs cuda:0

}


// Tensor Indexing
torch::Tensor num_y = torch::arange(0,36).reshape({6,6});

auto elem = num_y[3][3];                         // Specifies what value is present at position 3,3
std::cout << "Specifying finding with num_y[3][3] : " << elem << std::endl;
auto elem2 = num_y.index({3,3});                 // Returns the same thing
std::cout << "Specifying finding with num_y.index({3,3}) : " << elem2 << std::endl;



// Tensor Slicing with torch::indexing. I've already written using torch::indexing;
auto row0 = num_y.index({0, Slice()});           // First row
std::cout << "Obtaining the entire first row of num_y : " << row0 << std::endl;

auto col2 = num_y.index({Slice(), 2});           // Third column
std::cout << "Obtaining the entire third column of num_y : " << col2 << std::endl;

auto sub = num_y.index({Slice(0,2), Slice(1,3)});// Rows 0-1, Cols 1-2
std::cout << "Obtaining sub-rows and sub-columns of num_y : " << sub << std::endl;

auto step = num_y.index({Slice(), Slice(None, None, 2)}); // Every 2nd column, as in python we would do t(::2)
std::cout << "Obtaining every 2nd column of num_y : " << step << std::endl;


// Handling Mathematical Operations with Tensors
auto t_x = torch::tensor({1.0, 2.0, 3.0});
auto t_y = torch::tensor({4.0, 5.0, 6.0});

std::cout << "\n------------FROM HERE WE WILL SHOW COUPLE OF MATHEMATICAL OPERATIONS!--------------------" << std::endl;
auto add = t_x + t_y;
std::cout << "\nPrinting out sum of t_x and t_y : " << add << std::endl;
auto substraction = t_y - t_x;
std::cout << "\nPrinting out difference of t_x and t_y : " << substraction << std::endl;
auto mul = t_x * t_y;
std::cout << "\nPrinting out product of t_x and t_y : " << mul << std::endl;
auto div = t_y / t_x;
std::cout << "\nPrinting out quotient of t_x and t_y : " << div << std::endl;
auto pow = torch::pow(t_x, 2);
std::cout << "\nPrinting out power operation done on elements in t_x : " << pow << std::endl;
auto sq = torch::sqrt(t_y);
std::cout << "\nPrinting out square-root of elements in t_y : " << sq << std::endl;
auto exp = torch::exp(t_x);
std::cout << "\nPrinting out exponents of elements of t_x : " << exp << std::endl;
auto log = torch::log(t_y);
std::cout << "\nPrinting out natural logarithms of elements in t_y : " << log << std::endl;

    return 0;
}