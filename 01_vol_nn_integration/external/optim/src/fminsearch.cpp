// #include "dfo/fminsearch.h"
// #include <torch/torch.h>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <iostream>
// #include <iomanip>

// namespace dfo {

// // Helper function to check if convergence criteria are met exactly as in MATLAB
// bool checkConvergence(
//     const torch::Tensor& v,
//     const torch::Tensor& fv,
//     double tolX,
//     double tolFun
// ) {
//     // Get the best point (first column of v)
//     auto v1 = v.index({torch::indexing::Slice(), 0});
    
//     // MATLAB convergence check:
//     // if all(abs(fv(1)-fv(2:np1)) <= max(tolf,10*eps(fv(1)))) && ...
//     //    all(abs(v(:,2:np1)-v(:,1)) <= max(tolx,10*eps(max(v(:,1)))),'all')
    
//     // Check function value differences
//     auto fv0 = fv[0];
//     auto fDiffs = (fv.slice(0, 1) - fv0).abs();
//     auto tolFunEps = std::max(tolFun, 10.0 * std::numeric_limits<double>::epsilon() * std::abs(fv0.item<double>()));
//     bool fConverged = (fDiffs <= tolFunEps).all().item<bool>();
    
//     // Check coordinate differences
//     auto maxV1 = v1.abs().max().item<double>();
//     auto tolXEps = std::max(tolX, 10.0 * std::numeric_limits<double>::epsilon() * maxV1);
    
//     bool xConverged = true;
//     for (int64_t i = 1; i < v.size(1); ++i) {
//         auto vi = v.index({torch::indexing::Slice(), i});
//         auto diffs = (vi - v1).abs();
//         if ((diffs > tolXEps).any().item<bool>()) {
//             xConverged = false;
//             break;
//         }
//     }
    
//     // Both criteria must be met for convergence
//     return fConverged && xConverged;
// }

// FminsearchResult fminsearch(
//     const std::function<torch::Tensor(const torch::Tensor&)>& func,
//     const torch::Tensor& x0,
//     const FminsearchOptions& options
// ) {
//     // Initialize result
//     FminsearchResult result;
//     result.algorithm = "Nelder-Mead simplex direct search";
    
//     // Get the number of variables
//     auto originalShape = x0.sizes();
//     auto xin = x0.flatten();
//     int64_t n = xin.numel();
    
//     // Set default options based on dimension if needed
//     auto opts = options;
//     if (opts.maxIter == 200 && opts.maxFunEvals == 200) {
//         opts.setDefaultsFromDimension(n);
//     }
    
//     // Initialize parameters for Nelder-Mead
//     const double rho = 1.0;     // Reflection parameter
//     const double chi = 2.0;     // Expansion parameter
//     const double psi = 0.5;     // Contraction parameter
//     const double sigma = 0.5;   // Shrink parameter
    
//     // Number of points in the simplex (n+1)
//     int64_t np1 = n + 1;
    
//     // Initialize the simplex
//     auto v = torch::zeros({n, np1}, x0.options());
//     auto fv = torch::zeros({np1}, x0.options());
    
//     // Place initial guess in the simplex
//     v.index_put_({torch::indexing::Slice(), 0}, xin);
    
//     // Evaluate function at initial point
//     auto x = xin.clone().reshape(originalShape);
//     fv[0] = func(x);
//     int funcEvals = 1;
    
//     // Set up the initial simplex
//     const double usual_delta = 0.05;       // 5% deltas for non-zero terms
//     const double zero_term_delta = 0.00025; // Even smaller delta for zero elements
    
//     for (int64_t j = 0; j < n; ++j) {
//         auto y = xin.clone();
//         if (y[j].item<double>() != 0) {
//             y[j] = (1.0 + usual_delta) * y[j];
//         } else {
//             y[j] = zero_term_delta;
//         }
        
//         v.index_put_({torch::indexing::Slice(), j+1}, y);
//         x = y.clone().reshape(originalShape);
//         fv[j+1] = func(x);
//     }
    
//     funcEvals += n;
    
//     // Sort so v(:,1) has the lowest function value using tensor operations
//     auto sorted_indices = torch::argsort(fv);
//     auto newFv = torch::zeros_like(fv);
//     auto newV = torch::zeros_like(v);
    
//     for (int64_t i = 0; i < np1; ++i) {
//         auto idx = sorted_indices[i].item<int64_t>();
//         newFv[i] = fv[idx];
//         newV.index_put_({torch::indexing::Slice(), i}, v.index({torch::indexing::Slice(), idx}));
//     }
    
//     fv = newFv;
//     v = newV;
    
//     int itercount = 1;
//     std::string how = "initial simplex";
    
//     // Display initial information if requested
//     if (opts.display) {
//         std::cout << "Iteration   Func-count     f(x)         Procedure" << std::endl;
//         std::cout << std::setw(5) << itercount << std::setw(15) << funcEvals 
//                   << std::setw(15) << std::setprecision(6) << fv[0].item<double>() 
//                   << "   " << how << std::endl;
//     }
    
//     // Main algorithm loop
//     while (funcEvals < opts.maxFunEvals && itercount < opts.maxIter) {
//         // Check for convergence
//         if (checkConvergence(v, fv, opts.tolX, opts.tolFun)) {
//             break;
//         }
        
//         // Compute the reflection point
//         // xbar = average of the n (NOT n+1) best points
//         auto xbar = torch::mean(v.index({torch::indexing::Slice(), torch::indexing::Slice(0, n)}), 1);
//         auto xr = (1.0 + rho) * xbar - rho * v.index({torch::indexing::Slice(), n});
        
//         x = xr.clone().reshape(originalShape);
//         auto fxr = func(x);
//         funcEvals++;
        
//         if (fxr.item<double>() < fv[0].item<double>()) {
//             // Calculate the expansion point
//             auto xe = (1.0 + rho * chi) * xbar - rho * chi * v.index({torch::indexing::Slice(), n});
//             x = xe.clone().reshape(originalShape);
//             auto fxe = func(x);
//             funcEvals++;
            
//             if (fxe.item<double>() < fxr.item<double>()) {
//                 v.index_put_({torch::indexing::Slice(), n}, xe);
//                 fv[n] = fxe;
//                 how = "expand";
//             } else {
//                 v.index_put_({torch::indexing::Slice(), n}, xr);
//                 fv[n] = fxr;
//                 how = "reflect";
//             }
//         } else {
//             // fv[0] <= fxr
//             if (fxr.item<double>() < fv[n-1].item<double>()) {
//                 v.index_put_({torch::indexing::Slice(), n}, xr);
//                 fv[n] = fxr;
//                 how = "reflect";
//             } else {
//                 // fxr >= fv[n-1]
//                 // Perform contraction
//                 if (fxr.item<double>() < fv[n].item<double>()) {
//                     // Perform an outside contraction
//                     auto xc = (1.0 + psi * rho) * xbar - psi * rho * v.index({torch::indexing::Slice(), n});
//                     x = xc.clone().reshape(originalShape);
//                     auto fxc = func(x);
//                     funcEvals++;
                    
//                     if (fxc.item<double>() <= fxr.item<double>()) {
//                         v.index_put_({torch::indexing::Slice(), n}, xc);
//                         fv[n] = fxc;
//                         how = "contract outside";
//                     } else {
//                         // Perform a shrink
//                         how = "shrink";
//                     }
//                 } else {
//                     // Perform an inside contraction
//                     auto xcc = (1.0 - psi) * xbar + psi * v.index({torch::indexing::Slice(), n});
//                     x = xcc.clone().reshape(originalShape);
//                     auto fxcc = func(x);
//                     funcEvals++;
                    
//                     if (fxcc.item<double>() < fv[n].item<double>()) {
//                         v.index_put_({torch::indexing::Slice(), n}, xcc);
//                         fv[n] = fxcc;
//                         how = "contract inside";
//                     } else {
//                         // Perform a shrink
//                         how = "shrink";
//                     }
//                 }
                
//                 if (how == "shrink") {
//                     for (int64_t j = 1; j < np1; ++j) {
//                         auto vj = v.index({torch::indexing::Slice(), 0}) + 
//                                   sigma * (v.index({torch::indexing::Slice(), j}) - v.index({torch::indexing::Slice(), 0}));
//                         v.index_put_({torch::indexing::Slice(), j}, vj);
//                         x = vj.clone().reshape(originalShape);
//                         fv[j] = func(x);
//                     }
//                     funcEvals += n;
//                 }
//             }
//         }
        
//         // Sort the simplex using tensor operations
//         auto sorted_indices = torch::argsort(fv);
//         auto newFv = torch::zeros_like(fv);
//         auto newV = torch::zeros_like(v);
        
//         for (int64_t i = 0; i < np1; ++i) {
//             auto idx = sorted_indices[i].item<int64_t>();
//             newFv[i] = fv[idx];
//             newV.index_put_({torch::indexing::Slice(), i}, v.index({torch::indexing::Slice(), idx}));
//         }
        
//         fv = newFv;
//         v = newV;
        
//         itercount++;
        
//         // Display progress if requested
//         if (opts.display) {
//             std::cout << std::setw(5) << itercount << std::setw(15) << funcEvals 
//                       << std::setw(15) << std::setprecision(6) << fv[0].item<double>() 
//                       << "   " << how << std::endl;
//         }
//     }
    
//     // Set the result
//     result.x = v.index({torch::indexing::Slice(), 0}).reshape(originalShape);
//     result.fval = fv[0];
//     result.iterations = itercount;
//     result.funcCount = funcEvals;
    
//     // Set exit flag and message
//     if (funcEvals >= opts.maxFunEvals) {
//         result.exitflag = 0;
//         result.message = "Maximum number of function evaluations exceeded";
//     } else if (itercount >= opts.maxIter) {
//         result.exitflag = 0;
//         result.message = "Maximum number of iterations exceeded";
//     } else {
//         result.exitflag = 1;
//         result.message = "Optimization terminated: the current x satisfies the termination criteria";
//     }
    
//     return result;
// }

// } // namespace dfo
