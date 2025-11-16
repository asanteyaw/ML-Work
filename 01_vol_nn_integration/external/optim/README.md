# Derivative-Free Optimization (DFO) with LibTorch

This project implements derivative-free optimization algorithms in C++ using LibTorch (PyTorch's C++ API). The implementations are based on MATLAB's `fminsearch` and `patternsearch` functions.

## Features

- Pure tensor-based implementation using LibTorch
- Matches MATLAB's algorithms and results
- Supports arbitrary dimensions and tensor shapes
- Configurable optimization parameters
- Detailed output information
- Support for various constraint types (bounds, linear, nonlinear)

## Algorithms

### Nelder-Mead (fminsearch)

The Nelder-Mead simplex algorithm is a direct search method that does not use numerical or analytic gradients. The algorithm works by:

1. Creating an initial simplex around the starting point
2. Iteratively updating the simplex through reflection, expansion, contraction, and shrinking operations
3. Terminating when convergence criteria are met or maximum iterations/function evaluations are reached

### Pattern Search (patternsearch)

The Pattern Search algorithm is a direct search method that explores points in a mesh around the current point. The algorithm works by:

1. Defining a mesh around the current point
2. Polling points in the mesh to find a better solution
3. Adapting the mesh size based on success or failure
4. Terminating when convergence criteria are met or maximum iterations/function evaluations are reached

Pattern Search supports various constraint types:
- Bound constraints (lower and upper bounds)
- Linear constraints (equalities and inequalities)
- Nonlinear constraints (equalities and inequalities)

## Usage

### fminsearch

```cpp
#include "dfo/fminsearch.h"
#include <torch/torch.h>

// Define your objective function
torch::Tensor rosenbrock(const torch::Tensor& x) {
    auto x1 = x[0];
    auto x2 = x[1];
    return 100.0 * torch::pow(x2 - x1*x1, 2) + torch::pow(1.0 - x1, 2);
}

int main() {
    // Set initial point
    auto x0 = torch::tensor({-1.2, 1.0});
    
    // Set options (optional)
    dfo::FminsearchOptions options;
    options.display = true;  // Show iteration progress
    options.tolX = 1e-4;     // Tolerance on x
    options.tolFun = 1e-4;   // Tolerance on function value
    
    // Run optimization
    auto result = dfo::fminsearch(rosenbrock, x0, options);
    
    // Print results
    std::cout << "Solution: " << result.x << std::endl;
    std::cout << "Function value: " << result.fval.item<double>() << std::endl;
    std::cout << "Exit flag: " << result.exitflag << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Function evaluations: " << result.funcCount << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    
    return 0;
}
```

### patternsearch

```cpp
#include "dfo/patternsearch.h"
#include <torch/torch.h>

// Define your objective function
torch::Tensor objective(const torch::Tensor& x) {
    auto x1 = x[0];
    auto x2 = x[1];
    return torch::exp(-x1*x1 - x2*x2) * (1.0 + 5.0*x1 + 6.0*x2 + 12.0*x1*torch::cos(x2));
}

// Define nonlinear constraint function (optional)
std::tuple<torch::Tensor, torch::Tensor> nonlinear_constraint(const torch::Tensor& x) {
    auto x1 = x[0];
    auto x2 = x[1];
    
    // Inequality constraint: c(x) <= 0
    auto c = x1*x2/2.0 + torch::pow(x1+2.0, 2) + torch::pow(x2-2.0, 2)/2.0 - 2.0;
    c = c.reshape({1}); // Reshape to a 1D tensor with 1 element
    
    // No equality constraints
    auto ceq = torch::empty({0}, x.options());
    
    return {c, ceq};
}

int main() {
    // Set initial point
    auto x0 = torch::tensor({-2.0, -2.0});
    
    // Set options (optional)
    dfo::PatternsearchOptions options;
    options.display = true;  // Show iteration progress
    options.tolMesh = 1e-6;  // Tolerance on mesh size
    options.tolCon = 1e-6;   // Tolerance on constraints
    
    // Linear constraints (optional)
    auto A = torch::tensor({{-3.0, -2.0}, {-4.0, -7.0}});
    auto b = torch::tensor({-1.0, -8.0});
    
    // Bounds (optional)
    auto lb = torch::tensor({-10.0, -10.0});
    auto ub = torch::tensor({10.0, 10.0});
    
    // Run optimization with nonlinear constraints
    auto result = dfo::patternsearch(objective, x0, A, b, 
                                    torch::Tensor(), torch::Tensor(), // No equality constraints
                                    lb, ub, nonlinear_constraint, options);
    
    // Print results
    std::cout << "Solution: " << result.x << std::endl;
    std::cout << "Function value: " << result.fval.item<double>() << std::endl;
    std::cout << "Exit flag: " << result.exitflag << std::endl;
    std::cout << "Iterations: " << result.iterations << std::endl;
    std::cout << "Function evaluations: " << result.funcCount << std::endl;
    std::cout << "Problem type: " << result.problemType << std::endl;
    std::cout << "Message: " << result.message << std::endl;
    
    return 0;
}
```

## Examples

The implementation has been tested with examples from MATLAB's documentation:

### fminsearch examples

1. **Rosenbrock's function**: `f(x) = 100*(x(2) - x(1)^2)^2 + (1 - x(1))^2`
   - Solution: [1.0000, 1.0001]
   - Function value: 1.17736e-09

2. **objectivefcn1 function**: `f(x) = sum(exp(-(x(1)-x(2))^2 - 2*x(1)^2)*cos(x(2))*sin(2*x(2)))`
   - Solution: [-0.1696, -0.5086]
   - Function value: -13.131

3. **Parameterized Rosenbrock function**: `f(x,a) = 100*(x(2) - x(1)^2)^2 + (a-x(1))^2`
   - Solution: [3.0000, 9.0000] (with a=3)
   - Function value: 2.78533e-12

4. **Three-variable function**: `f(x) = -norm(x+x0)^2*exp(-norm(x-x0)^2 + sum(x))`
   - Solution: [1.5361, 2.5645, 3.5933] (with x0=[1,2,3])
   - Function value: -59564.6

### patternsearch examples

1. **Unconstrained minimization**: `f(x) = exp(-x(1)^2-x(2)^2)*(1+5*x(1) + 6*x(2) + 12*x(1)*cos(x(2)))`
   - Solution: [-0.7037, -0.1860]
   - Function value: -7.0254

2. **Bound constrained minimization**: Same function with bounds `0 <= x(1)` and `x(2) <= -3`
   - Solution: [0.1880, -3.0000]
   - Function value: -0.5095

3. **Linear inequality constrained minimization**: Same function with constraints `-3*x(1) - 2*x(2) <= -1` and `-4*x(1) - 7*x(2) <= -8`
   - Solution: [5.2827, -1.8758]
   - Function value: -0.0000

4. **Nonlinear constrained minimization**: Same function with nonlinear constraint `x(1)*x(2)/2 + (x(1)+2)^2 + (x(2)-2)^2/2 - 2 <= 0`
   - Solution: [-1.5144, 0.0875]
   - Function value: -5.5917


## Building

The project uses CMake for building. Make sure you have LibTorch installed.

```bash
mkdir build
cd build
cmake ..
make
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
