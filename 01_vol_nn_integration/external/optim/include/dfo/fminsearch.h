#pragma once

#include <torch/torch.h>
#include <functional>
#include <tuple>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <cmath>

namespace dfo {

struct FminsearchOptions {
    int maxIter = 200;          // Maximum number of iterations
    int maxFunEvals = 200;      // Maximum number of function evaluations
    double tolX = 1e-4;         // Tolerance on x (exactly as in MATLAB)
    double tolFun = 1e-4;       // Tolerance on function value (exactly as in MATLAB)
    bool display = false;       // Display optimization progress
    
    // Set default options based on number of variables
    void setDefaultsFromDimension(int64_t n) {
        maxIter = 200 * n;
        maxFunEvals = 200 * n;
    }
};

struct FminsearchResult {
    torch::Tensor x;            // Solution
    torch::Tensor fval;         // Function value at solution (scalar tensor)
    int exitflag;               // Exit flag (1: converged, 0: max iter/evals, -1: terminated)
    int iterations;             // Number of iterations
    int funcCount;              // Number of function evaluations
    std::string message;        // Exit message
    std::string algorithm;      // Algorithm name
};

namespace detail {
    // Helper function to safely extract a value from a tensor
    // Throws a more informative error if the tensor is not a scalar
    template<typename T>
    inline T safe_item(const torch::Tensor& tensor, const std::string& context) {
        if (tensor.numel() != 1) {
            std::ostringstream oss;
            oss << "Error in " << context << ": Cannot convert a tensor with " 
                << tensor.numel() << " elements to a scalar. Tensor sizes: " 
                << tensor.sizes() << ", tensor values: " << tensor;
            throw std::runtime_error(oss.str());
        }
        return tensor.item<T>();
    }
    
    // Convenience function for double values (most common case)
    inline double safe_item(const torch::Tensor& tensor, const std::string& context) {
        return safe_item<double>(tensor, context);
    }
    
    // Helper function to check if convergence criteria are met exactly as in MATLAB
    // This implementation exactly matches MATLAB's behavior
    inline bool checkConvergence(
        const torch::Tensor& v,
        const torch::Tensor& fv,
        double tolX,
        double tolFun
    ) {
        // Get the best point (first column of v)
        auto v1 = v.select(1, 0);
        
        // MATLAB convergence check:
        // if all(abs(fv(1)-fv(2:np1)) <= max(tolf,10*eps(fv(1)))) && ...
        //    all(abs(v(:,2:np1)-v(:,1)) <= max(tolx,10*eps(max(v(:,1)))),'all')
        
        // Get the function value at the best point
        auto fv0 = safe_item(fv[0], "checkConvergence - fv[0]");
        
        // Calculate tolerance for function values
        auto tolFunEps = std::max(tolFun, 10.0 * std::numeric_limits<double>::epsilon() * std::abs(fv0));
        
        // Check function value differences - exactly as in MATLAB
        bool fConverged = true;
        for (int64_t i = 1; i < fv.size(0); ++i) {
            if (std::abs(safe_item(fv[i], "checkConvergence - fv[i]") - fv0) > tolFunEps) {
                fConverged = false;
                break;
            }
        }
        
        if (!fConverged) return false;
        
        // Find the maximum absolute value in the best point
        double maxV1 = 0.0;
        for (int64_t i = 0; i < v1.size(0); ++i) {
            maxV1 = std::max(maxV1, std::abs(safe_item(v1[i], "checkConvergence - v1[i]")));
        }
        
        // Calculate tolerance for coordinate differences
        auto tolXEps = std::max(tolX, 10.0 * std::numeric_limits<double>::epsilon() * maxV1);
        
        // Check coordinate differences - exactly as in MATLAB
        bool xConverged = true;
        for (int64_t j = 1; j < v.size(1); ++j) {
            auto vj = v.select(1, j);
            for (int64_t i = 0; i < v1.size(0); ++i) {
                if (std::abs(safe_item(vj[i], "checkConvergence - vj[i]") - safe_item(v1[i], "checkConvergence - v1[i] comparison")) > tolXEps) {
                    xConverged = false;
                    break;
                }
            }
            if (!xConverged) break;
        }
        
        // Both criteria must be met for convergence
        return fConverged && xConverged;
    }
}

/**
 * @brief Multidimensional unconstrained nonlinear minimization using Nelder-Mead method
 * 
 * @param func Function to minimize, takes a torch::Tensor and returns a torch::Tensor (scalar)
 * @param x0 Initial point
 * @param options Optimization options
 * @return FminsearchResult containing the solution and optimization information
 */
inline FminsearchResult fminsearch(
    const std::function<torch::Tensor(const torch::Tensor&)>& func,
    const torch::Tensor& x0,
    const FminsearchOptions& options = FminsearchOptions()
) {
    // Initialize result
    FminsearchResult result;
    result.algorithm = "Nelder-Mead simplex direct search";
    
    // Get the number of variables
    auto originalShape = x0.sizes();
    auto xin = x0.flatten();
    int64_t n = xin.numel();
    
    // Set default options based on dimension if needed
    auto opts = options;
    if (opts.maxIter == 200 && opts.maxFunEvals == 200) {
        opts.setDefaultsFromDimension(n);
    }
    
    // Initialize parameters for Nelder-Mead
    const double rho = 1.0;     // Reflection parameter
    const double chi = 2.0;     // Expansion parameter
    const double psi = 0.5;     // Contraction parameter
    const double sigma = 0.5;   // Shrink parameter
    
    // Number of points in the simplex (n+1)
    int64_t np1 = n + 1;
    
    // Initialize the simplex - use contiguous memory layout for better performance
    auto v = torch::zeros({n, np1}, x0.options()).contiguous();
    auto fv = torch::zeros({np1}, x0.options()).contiguous();
    
    // Place initial guess in the simplex
    v.select(1, 0).copy_(xin);
    
    // Evaluate function at initial point
    auto x = xin.clone().reshape(originalShape);
    
    // Use torch::NoGradGuard to disable gradient tracking during optimization
    torch::NoGradGuard no_grad;
    
    // Check device type for device-specific optimizations
    torch::DeviceType device_type = x0.device().type();
    bool is_mps = (device_type == torch::kMPS);
    bool is_cuda = (device_type == torch::kCUDA);
    bool is_gpu = is_mps || is_cuda;
    
    fv[0] = func(x);
    int funcEvals = 1;
    
    // Set up the initial simplex
    const double usual_delta = 0.05;       // 5% deltas for non-zero terms
    const double zero_term_delta = 0.00025; // Even smaller delta for zero elements
    
    // Create the simplex in a vectorized way
    for (int64_t j = 0; j < n; ++j) {
        auto y = xin.clone();
        if (detail::safe_item(y[j], "initial simplex setup - y[j]") != 0) {
            y[j] = (1.0 + usual_delta) * y[j];
        } else {
            y[j] = zero_term_delta;
        }
        
        v.select(1, j+1).copy_(y);
        x = y.clone().reshape(originalShape);
        fv[j+1] = func(x);
    }
    
    funcEvals += n;
    
    // Sort so v(:,1) has the lowest function value using argsort
    // This matches MATLAB's sort behavior exactly
    auto sorted_indices = torch::argsort(fv);
    
    // Use index_select for more efficient reordering
    auto fv_new = fv.index_select(0, sorted_indices);
    
    // Reorder columns of v to match MATLAB's behavior exactly
    auto v_new = torch::zeros_like(v);
    for (int64_t i = 0; i < np1; ++i) {
        v_new.select(1, i).copy_(v.select(1, detail::safe_item<int64_t>(sorted_indices[i], "sorted_indices[i] in main loop reordering")));
    }
    
    // Update the tensors
    fv = fv_new;
    v = v_new;
    
    int itercount = 1;
    std::string how = "initial simplex";
    
    // Display initial information if requested
    if (opts.display) {
        std::cout << "Iteration   Func-count     f(x)         Procedure" << std::endl;
        std::cout << std::setw(5) << itercount << std::setw(15) << funcEvals 
                  << std::setw(15) << std::setprecision(6) << detail::safe_item(fv[0], "display initial - fv[0]") 
                  << "   " << how << std::endl;
    }
    
    // Pre-allocate tensors for repeated use to avoid memory allocations in the loop
    auto xbar = torch::zeros({n}, x0.options());
    auto xr = torch::zeros({n}, x0.options());
    auto xe = torch::zeros({n}, x0.options());
    auto xc = torch::zeros({n}, x0.options());
    auto xcc = torch::zeros({n}, x0.options());
    
    // Main algorithm loop
    while (funcEvals < opts.maxFunEvals && itercount < opts.maxIter) {
        // Check for convergence
        if (detail::checkConvergence(v, fv, opts.tolX, opts.tolFun)) {
            break;
        }
        
        // Compute the reflection point
        // xbar = average of the n (NOT n+1) best points
        // Use mean along dimension 1 for vectorized computation
        xbar.copy_(v.slice(1, 0, n).mean(1));
        
        // Compute reflection point: xr = (1 + rho) * xbar - rho * v(:,end)
        xr.copy_((1.0 + rho) * xbar - rho * v.select(1, n));
        
        x = xr.clone().reshape(originalShape);
        auto fxr = func(x);
        funcEvals++;
        
        if (detail::safe_item(fxr, "fxr comparison") < detail::safe_item(fv[0], "fv[0] comparison")) {
            // Calculate the expansion point
            xe.copy_((1.0 + rho * chi) * xbar - rho * chi * v.select(1, n));
            x = xe.clone().reshape(originalShape);
            auto fxe = func(x);
            funcEvals++;
            
            if (detail::safe_item(fxe, "fxe comparison") < detail::safe_item(fxr, "fxr in expansion")) {
                v.select(1, n).copy_(xe);
                fv[n] = fxe;
                how = "expand";
            } else {
                v.select(1, n).copy_(xr);
                fv[n] = fxr;
                how = "reflect";
            }
        } else {
            // fv[0] <= fxr
            if (detail::safe_item(fxr, "fxr in reflect") < detail::safe_item(fv[n-1], "fv[n-1] comparison")) {
                v.select(1, n).copy_(xr);
                fv[n] = fxr;
                how = "reflect";
            } else {
                // fxr >= fv[n-1]
                // Perform contraction
                if (detail::safe_item(fxr, "fxr in contraction") < detail::safe_item(fv[n], "fv[n] comparison")) {
                    // Perform an outside contraction
                    xc.copy_((1.0 + psi * rho) * xbar - psi * rho * v.select(1, n));
                    x = xc.clone().reshape(originalShape);
                    auto fxc = func(x);
                    funcEvals++;
                    
                    if (detail::safe_item(fxc, "fxc comparison") <= detail::safe_item(fxr, "fxr in outside contraction")) {
                        v.select(1, n).copy_(xc);
                        fv[n] = fxc;
                        how = "contract outside";
                    } else {
                        // Perform a shrink
                        how = "shrink";
                    }
                } else {
                    // Perform an inside contraction
                    xcc.copy_((1.0 - psi) * xbar + psi * v.select(1, n));
                    x = xcc.clone().reshape(originalShape);
                    auto fxcc = func(x);
                    funcEvals++;
                    
                    if (detail::safe_item(fxcc, "fxcc comparison") < detail::safe_item(fv[n], "fv[n] in inside contraction")) {
                        v.select(1, n).copy_(xcc);
                        fv[n] = fxcc;
                        how = "contract inside";
                    } else {
                        // Perform a shrink
                        how = "shrink";
                    }
                }
                
                if (how == "shrink") {
                    // Highly optimized shrink operation
                    auto v0 = v.select(1, 0);
                    
                    // Batch process the shrink operation for better memory efficiency
                    // First, compute all new points in a vectorized way
                    for (int64_t j = 1; j < np1; ++j) {
                        // v(:,j) = v(:,1) + sigma * (v(:,j) - v(:,1))
                        v.select(1, j).copy_(v0 + sigma * (v.select(1, j) - v0));
                    }
                    
                    // Then evaluate the function at each point
                    // This separation allows for better memory management
                    for (int64_t j = 1; j < np1; ++j) {
                        x = v.select(1, j).clone().reshape(originalShape);
                        fv[j] = func(x);
                    }
                    
                    funcEvals += n;
                }
            }
        }
        
    // Sort the simplex using argsort - exactly matching MATLAB's behavior
    sorted_indices = torch::argsort(fv);
    
    // Use index_select for more efficient reordering
    auto fv_new = fv.index_select(0, sorted_indices);
    
    // Reorder columns of v to match MATLAB's behavior exactly
    auto v_new = torch::zeros_like(v);
    for (int64_t i = 0; i < np1; ++i) {
        v_new.select(1, i).copy_(v.select(1, detail::safe_item<int64_t>(sorted_indices[i], "sorted_indices[i] in reordering")));
    }
    
    // Update the tensors
    fv = fv_new;
    v = v_new;
        
        itercount++;
        
        // Display progress if requested
        if (opts.display) {
            std::cout << std::setw(5) << itercount << std::setw(15) << funcEvals 
                      << std::setw(15) << std::setprecision(6) << detail::safe_item(fv[0], "display progress - fv[0]") 
                      << "   " << how << std::endl;
        }
        
        // Explicitly free any temporary tensors
        if (itercount % 10 == 0) {
            // Force device synchronization and garbage collection every 10 iterations
            if (is_gpu) {
                try {
                    // For CUDA devices
                    if (is_cuda) {
                        torch::cuda::synchronize();
                    }
                    // For MPS devices
                    else if (is_mps) {
                        // MPS doesn't have an explicit synchronize method like CUDA
                        // But we can force a synchronization by performing a small operation
                        auto dummy = torch::ones({1}, x0.options());
                        dummy.add_(1.0);
                    }
                } catch (const c10::Error& e) {
                    // Catch and handle any device-related errors
                    if (opts.display) {
                        std::cerr << "Warning: Device synchronization failed: " << e.what() << std::endl;
                        std::cerr << "Continuing optimization without synchronization." << std::endl;
                    }
                }
            }
            
            // Memory management is handled by the device runtime
            // No explicit cache clearing needed
        }
    }
    
    // Set the result
    result.x = v.select(1, 0).reshape(originalShape);
    result.fval = fv[0];
    result.iterations = itercount;
    result.funcCount = funcEvals;
    
    // Set exit flag and message
    if (funcEvals >= opts.maxFunEvals) {
        result.exitflag = 0;
        result.message = "Maximum number of function evaluations exceeded";
    } else if (itercount >= opts.maxIter) {
        result.exitflag = 0;
        result.message = "Maximum number of iterations exceeded";
    } else {
        result.exitflag = 1;
        result.message = "Optimization terminated: the current x satisfies the termination criteria";
    }
    
    return result;
}

} // namespace dfo
