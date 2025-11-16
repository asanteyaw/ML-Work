#include "dfo/fminsearch.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

// Example 1: Rosenbrock's function
// f(x) = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2
torch::Tensor rosenbrock(const torch::Tensor& x) {
    auto x1 = x[0];
    auto x2 = x[1];
    return 100.0 * torch::pow(x2 - x1*x1, 2) + torch::pow(1.0 - x1, 2);
}

// Example 2: objectivefcn1 function
// f(x) = sum(exp(-(x(1)-x(2))^2 - 2*x(1)^2)*cos(x(2))*sin(2*x(2))) for k=-10:10
torch::Tensor objectivefcn1(const torch::Tensor& x) {
    auto x1 = x[0];
    auto x2 = x[1];
    
    // Calculate the expression once (the k value doesn't actually affect the calculation in the MATLAB code)
    auto term = torch::exp(-torch::pow(x1-x2, 2) - 2.0*x1*x1) * torch::cos(x2) * torch::sin(2.0*x2);
    
    // Multiply by 21 (the number of iterations in the loop from k=-10 to k=10)
    return term * 21.0;
}

// Example 3: Parameterized function
// f(x,a) = 100*(x(2) - x(1)^2)^2 + (a-x(1))^2
torch::Tensor parameterized_rosenbrock(const torch::Tensor& x, double a) {
    auto x1 = x[0];
    auto x2 = x[1];
    return 100.0 * torch::pow(x2 - x1*x1, 2) + torch::pow(a - x1, 2);
}

// Example 4: Three-variable function
// f(x) = -norm(x+x0)^2*exp(-norm(x-x0)^2 + sum(x))
torch::Tensor three_var_func(const torch::Tensor& x, const torch::Tensor& x0) {
    auto sum_x = x.sum();
    auto norm_plus = (x + x0).norm();
    auto norm_minus = (x - x0).norm();
    
    return -torch::pow(norm_plus, 2) * torch::exp(-torch::pow(norm_minus, 2) + sum_x);
}

// Example 5: High-dimensional function for memory efficiency testing
// f(x) = sum(x^2) + 0.1*sum(x)^2
torch::Tensor high_dim_func(const torch::Tensor& x) {
    auto sum_squared = torch::sum(torch::pow(x, 2));
    auto squared_sum = torch::pow(torch::sum(x), 2);
    return sum_squared + 0.1 * squared_sum;
}

// Example 6: Function with tensor operations that could cause memory issues
// f(x) = sum(exp(-x^2)) + norm(x)
torch::Tensor memory_intensive_func(const torch::Tensor& x) {
    auto exp_term = torch::sum(torch::exp(-torch::pow(x, 2)));
    auto norm_term = x.norm();
    return exp_term + norm_term;
}

// Helper function to print results
void print_result(const std::string& name, const dfo::FminsearchResult& result) {
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "Solution: " << result.x << std::endl;
    std::cout << "Function value: " << result.fval.item<double>() << std::endl;
    std::cout << "Exit flag: " << result.exitflag << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Function evaluations: " << result.funcCount << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    std::cout << std::endl;
}

// Helper function to run a test and measure execution time
template<typename Func>
void run_timed_test(const std::string& name, Func func, const torch::Tensor& x0, 
                   const dfo::FminsearchOptions& options, torch::Device device) {
    std::cout << "\n\n--- " << name << " on " << device << " ---\n" << std::endl;
    
    try {
        // Move input tensor to the specified device
        torch::Tensor x0_device;
        try {
            x0_device = x0.to(device);
        } catch (const c10::Error& e) {
            std::cerr << "Error moving tensor to device " << device << ": " << e.what() << std::endl;
            std::cerr << "Falling back to CPU." << std::endl;
            device = torch::kCPU;
            x0_device = x0.to(device);
        }
        
        // Run the optimization
        auto start = std::chrono::high_resolution_clock::now();
        auto result = dfo::fminsearch(func, x0_device, options);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        
        print_result(name, result);
        std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error during optimization: " << e.what() << std::endl;
        std::cerr << "Test failed on device " << device << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        std::cerr << "Test failed on device " << device << std::endl;
    } catch (...) {
        std::cerr << "Unknown error occurred during test." << std::endl;
        std::cerr << "Test failed on device " << device << std::endl;
    }
}

int main() {
    // Set options
    dfo::FminsearchOptions options;
    options.display = true;
    
    // Determine available devices
    torch::Device cpu_device(torch::kCPU);
    torch::Device gpu_device = cpu_device;
    
    // Check for available devices in a safe way
    bool cuda_available = false;
    bool mps_available = false;
    
    try {
        cuda_available = torch::cuda::is_available();
    } catch (const c10::Error& e) {
        std::cout << "CUDA check failed: " << e.what() << std::endl;
        std::cout << "Will not test on CUDA." << std::endl;
    }
    
    try {
        mps_available = torch::mps::is_available();
    } catch (const c10::Error& e) {
        std::cout << "MPS check failed: " << e.what() << std::endl;
        std::cout << "Will not test on MPS." << std::endl;
    }
    
    // Set up the GPU device if available
    if (cuda_available) {
        try {
            gpu_device = torch::Device(torch::kCUDA);
            std::cout << "CUDA is available. Will test on GPU." << std::endl;
        } catch (const c10::Error& e) {
            std::cout << "Failed to set CUDA device: " << e.what() << std::endl;
            cuda_available = false;
        }
    } else if (mps_available) {
        try {
            gpu_device = torch::Device(torch::kMPS);
            std::cout << "MPS is available. Will test on Apple Silicon GPU." << std::endl;
        } catch (const c10::Error& e) {
            std::cout << "Failed to set MPS device: " << e.what() << std::endl;
            mps_available = false;
        }
    } else {
        std::cout << "No GPU available. Testing on CPU only." << std::endl;
    }
    
    std::cout << "Testing fminsearch implementation with examples from MATLAB documentation\n" << std::endl;
    
    // Example 1: Rosenbrock's function
    auto x0_rosenbrock = torch::tensor({-1.2, 1.0});
    run_timed_test("Rosenbrock's function", rosenbrock, x0_rosenbrock, options, cpu_device);
    
    // Example 2: objectivefcn1 function
    auto x0_obj1 = torch::tensor({0.25, -0.25});
    run_timed_test("objectivefcn1 function", objectivefcn1, x0_obj1, options, cpu_device);
    
    // Example 3: Parameterized function
    double a = 3.0;
    auto x0_param = torch::tensor({-1.0, 1.9});
    run_timed_test(
        "Parameterized Rosenbrock function",
        [a](const torch::Tensor& x) { return parameterized_rosenbrock(x, a); },
        x0_param,
        options,
        cpu_device
    );
    
    // Example 4: Three-variable function
    auto x0_three = torch::tensor({1.0, 2.0, 3.0});
    run_timed_test(
        "Three-variable function",
        [x0_three](const torch::Tensor& x) { return three_var_func(x, x0_three.to(x.device())); },
        x0_three,
        options,
        cpu_device
    );
    
    // Example 5: High-dimensional function (memory efficiency test)
    std::cout << "\n\n=== Memory Efficiency Tests ===\n" << std::endl;
    
    // Test with increasing dimensions to demonstrate memory efficiency
    for (int dim : {10, 20, 50}) {
        auto x0_high_dim = torch::ones({dim});
        
        dfo::FminsearchOptions high_dim_options;
        high_dim_options.display = false;  // Disable display for cleaner output
        high_dim_options.maxIter = 100;    // Limit iterations for faster testing
        
        std::string test_name = "High-dimensional function (dim=" + std::to_string(dim) + ")";
        run_timed_test(test_name, high_dim_func, x0_high_dim, high_dim_options, cpu_device);
        
        // If GPU is available, also test on GPU
        if (cuda_available || mps_available) {
            run_timed_test(test_name, high_dim_func, x0_high_dim, high_dim_options, gpu_device);
        }
    }
    
    // Example 6: Memory-intensive function
    std::cout << "\n\n=== Memory-Intensive Function Test ===\n" << std::endl;
    
    // Test with a function that performs operations that could cause memory issues
    auto x0_mem_intensive = torch::ones({30});
    
    dfo::FminsearchOptions mem_options;
    mem_options.display = true;
    mem_options.maxIter = 50;  // Limit iterations for faster testing
    
    run_timed_test("Memory-intensive function", memory_intensive_func, x0_mem_intensive, mem_options, cpu_device);
    
    // If GPU is available, also test on GPU
    if (cuda_available || mps_available) {
        run_timed_test("Memory-intensive function", memory_intensive_func, x0_mem_intensive, mem_options, gpu_device);
    }
    
    // Test numerical accuracy by comparing with known solutions
    std::cout << "\n\n=== Numerical Accuracy Tests ===\n" << std::endl;
    
    // Rosenbrock function has a known minimum at [1, 1] with value 0
    auto x0_accuracy = torch::tensor({0.0, 0.0});
    dfo::FminsearchOptions accuracy_options;
    accuracy_options.display = false;
    accuracy_options.tolX = 1e-6;
    accuracy_options.tolFun = 1e-6;
    
    // Function to run accuracy test on a specific device
    auto run_accuracy_test = [&](const std::string& device_name, torch::Device device) {
        std::cout << "\n--- Numerical Accuracy Test on " << device_name << " ---\n" << std::endl;
        
        try {
            // Move input tensor to the specified device
            torch::Tensor x0_device;
            try {
                x0_device = x0_accuracy.to(device);
            } catch (const c10::Error& e) {
                std::cerr << "Error moving tensor to device " << device << ": " << e.what() << std::endl;
                std::cerr << "Skipping accuracy test on " << device_name << std::endl;
                return;
            }
            
            // Run the optimization
            auto result = dfo::fminsearch(rosenbrock, x0_device, accuracy_options);
            
            // Expected solution
            auto expected_solution = torch::tensor({1.0, 1.0}).to(device);
            
            std::cout << "Rosenbrock function accuracy test:" << std::endl;
            std::cout << "Expected solution: " << expected_solution << std::endl;
            std::cout << "Actual solution: " << result.x << std::endl;
            std::cout << "Expected function value: 0.0" << std::endl;
            std::cout << "Actual function value: " << result.fval.item<double>() << std::endl;
            
            double solution_error = (result.x - expected_solution).norm().item<double>();
            std::cout << "Solution error: " << solution_error << std::endl;
            std::cout << "Test " << (solution_error < 1e-5 ? "PASSED" : "FAILED") << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error during accuracy test: " << e.what() << std::endl;
            std::cerr << "Test failed on device " << device_name << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception: " << e.what() << std::endl;
            std::cerr << "Test failed on device " << device_name << std::endl;
        } catch (...) {
            std::cerr << "Unknown error occurred during accuracy test." << std::endl;
            std::cerr << "Test failed on device " << device_name << std::endl;
        }
    };
    
    // Run accuracy tests on CPU
    run_accuracy_test("CPU", cpu_device);
    
    // If GPU is available, also test on GPU
    if (cuda_available || mps_available) {
        run_accuracy_test(cuda_available ? "CUDA" : "MPS", gpu_device);
    }
    
    return 0;
}
