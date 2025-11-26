// Functionals.h
#pragma once

#include <torch/torch.h>
#include <vector>

namespace losses {

// Huber Loss Function
torch::Tensor huber_loss(const torch::Tensor& predictions,
                         const torch::Tensor& targets,
                         const torch::Tensor& norm_values=torch::tensor(0.0),
                         double delta = 1.0,
                         const int64_t use_norm = 0)
{
    // Compute residuals and normalize

    auto errors = (1 - use_norm) * (predictions - targets) +
                           use_norm * ((predictions - targets) / (norm_values + 1e-8)); // Normalize residuals

    // Compute Huber loss
    auto abs_errors = torch::abs(errors);
    auto quadratic_mask = abs_errors <= delta; // Quadratic region condition
    auto quadratic_loss = 0.5 * torch::pow(errors, 2); // MSE region
    auto linear_loss = delta * abs_errors - 0.5 * delta * delta; // MAE region

    // Combine quadratic and linear loss
    auto loss = torch::where(quadratic_mask, quadratic_loss, linear_loss);

    // Return the mean or summed loss
    return loss.mean(); // Use .sum() for summed loss if required
}

torch::Tensor huber_per_batch(torch::Tensor tensor1, torch::Tensor tensor2, double delta = 0.055) {
    // Ensure tensors have the same shape
    TORCH_CHECK(tensor1.sizes() == tensor2.sizes(), "Tensors must have the same shape!");

    // Compute the absolute difference
    torch::Tensor abs_diff = (tensor1 - tensor2).abs();

    // Compute the squared loss for elements where abs_diff <= delta
    torch::Tensor squared_loss = 0.5 * (tensor1 - tensor2).pow(2);

    // Compute the linear loss for elements where abs_diff > delta
    torch::Tensor linear_loss = delta * (abs_diff - 0.5 * delta);

    // Combine the two cases using a mask
    torch::Tensor huber_loss = torch::where(abs_diff <= delta, squared_loss, linear_loss);

    // Compute mean Huber loss per batch
    return huber_loss.mean({1, 2}); // Reduce over dimensions 1 and 2 (columns and depth)
}

torch::Tensor mse_per_batch(torch::Tensor tensor1, torch::Tensor tensor2) {
    // Ensure tensors have the same shape
    TORCH_CHECK(tensor1.sizes() == tensor2.sizes(), "Tensors must have the same shape!");

    // Compute element-wise squared differences
    torch::Tensor squared_diff = (tensor1 - tensor2).pow(2);

    // Compute mean squared error per batch
    return squared_diff.mean({1, 2}); // Reduce over dimensions 1 and 2 (columns and depth)
}




torch::Tensor linex_loss(const torch::Tensor& predicted_variance, 
                                    const torch::Tensor& actual_variance, 
                                    double a, 
                                    double b) {
    // Compute the difference
    auto diff = predicted_variance - actual_variance;

    // Compute the LINEX loss
    auto exp_term = torch::exp(b * diff);
    auto linex_loss = a * (exp_term - b * diff - 1);

    // Return the mean loss
    return linex_loss.mean();
}

} // namespace losses

namespace transforms {

// Slope Transformation
torch::Tensor slope_transform(const torch::Tensor& params, 
                              const torch::Tensor& lb=torch::tensor({0.0,0.0,0.0,0.0,0.0}), 
                              const torch::Tensor& ub=torch::tensor({1.0,1.0,1.0,5.0,5.0}), 
                              torch::Tensor slope=torch::tensor(2.0)
                             ) 
{
    auto l = torch::zeros_like(params);
    auto u = torch::ones_like(params);
    u.index_put_({torch::indexing::Slice(), 3, 5}, torch::full({2}, 5.0));

    return torch::log((params - l) / (u - params)) / slope;
}

torch::Tensor slope_inverse_transform(const torch::Tensor& scaled_params, 
                                      const torch::Tensor& lb=torch::tensor({0.0,0.0,0.0,0.0,0.0}), 
                                      const torch::Tensor& ub=torch::tensor({1.0,1.0,1.0,5.0,5.0}), 
                                      torch::Tensor slope=torch::tensor(2.0)
                                     )
{
    auto l = torch::zeros_like(scaled_params);
    auto u = torch::ones_like(scaled_params);
    u.index_put_({torch::indexing::Slice(), 3, 5}, torch::full({2}, 5.0));

    return l + (u - l) / (1 + torch::exp(-slope * scaled_params));
}


torch::Tensor act_transform(const torch::Tensor& params) {

    const std::vector<std::pair<double, double>> bounds = {
        {0.000085, 0.0025}, {0.015, 0.089}, {0.75, 0.98}, {0.015, 0.1}, {0.2, 2.5}
    };

    // Validate dimensions
    TORCH_CHECK(params.size(1) == bounds.size(), "Number of columns must match the number of bounds pairs.");

    // Clone the input tensor to avoid modifying the original
    torch::Tensor out = torch::zeros_like(params);

    // Stack column-wise transformations
    for (size_t col = 0; col < bounds.size(); ++col) {
        double a = bounds[col].first, b = bounds[col].second;

        // Select the column
        auto column = params.select(1, col);

        // Apply the transformation
        auto mask = (column >= 0.0) & (column <= 1.0);

        // Notify if there's a violation
        if (~mask.all().item<bool>()) 
            std::cout << "Some values in column " << col << " are outside of 0 and 1" << std::endl;
        
        // Apply transformation and update output tensor
        //out.select(1, col) = a + (b - a) * torch::where(mask, column, torch::sigmoid(column));
        out.index_put_({torch::indexing::Slice(), static_cast<int64_t>(col)} , a + (b - a) * torch::where(mask, column, 
torch::sigmoid(column)));
    }

    return out;
}

}  // namespace transforms


namespace volatilities {

std::unordered_map<std::string, std::function<torch::Tensor(torch::TensorList, torch::Tensor&, const torch::Tensor&, torch::Tensor&)>> funcMap;

torch::Tensor ngarch_eq(const torch::Tensor& params, const torch::Tensor& x){
   int64_t T = x.size(1);  // x is returns

   torch::Tensor h_l{torch::empty_like(x)}, z_l{torch::empty_like(x)};
   torch::Tensor r{torch::tensor(0.019)}, d{torch::tensor(0.012)};

   auto omega_s = params.select(1, 0);
   auto alpha_s = params.select(1, 1);
   auto phi_s = params.select(1, 2);
   auto lamda_s = params.select(1, 3);
   auto gamma_s = params.select(1, 4);

   auto rd = r - d;
   torch::Tensor h_t = omega_s;
   torch::Tensor z_t = (x.select(1, 0) - rd + 0.5 * h_t - lamda_s * torch::sqrt(h_t)) / torch::sqrt(h_t);
   h_l.select(1, 0) = h_t;
   z_l.select(1, 0) = z_t;
   //h_l.index_put_({torch::indexing::Slice(), 0}, h_t);
   //z_l.index_put_({torch::indexing::Slice(), 0}, z_t);
   
   for(int64_t t{1}; t < T; ++t){
      auto hs = h_l.select(1, t-1);
      auto zs = z_l.select(1, t-1);
      h_t = omega_s + phi_s * (hs - omega_s) + alpha_s * hs * (zs.pow(2) - 2.0 * gamma_s * zs - 1.0);
      z_t = (x.select(1, t) - rd + 0.5 * h_t - lamda_s * torch::sqrt(h_t)) / torch::sqrt(h_t);

      h_l.select(1, t) = h_t;
      z_l.select(1, t) = z_t; 
      //h_l.index_put_({torch::indexing::Slice(), t}, h_t);
      //z_l.index_put_({torch::indexing::Slice(), t}, z_t);
   }
   
   return h_l;
}

torch::Tensor EGARCH(torch::TensorList params, torch::Tensor& z, const torch::Tensor& x, torch::Tensor& h) {
    if (params.size() != 9) {
        std::cerr << "Realized EGARCH requires 9 parameters!" << std::endl;
        std::abort();
    }
    torch::Tensor om = params[0];   // omega
    torch::Tensor t1 = params[1];   // tau1
    torch::Tensor t2 = params[2];   // tau2
    torch::Tensor g = params[3];    // gamma
    torch::Tensor b = params[4];    // beta
    torch::Tensor xi = params[5];    // xi
    torch::Tensor p = params[6];    // phi
    torch::Tensor d1 = params[7];    // delta1
    torch::Tensor d2 = params[8];    // delta2

    auto tau_z = t1*z+t2*(torch::pow(z,2)-1);
    auto del_z = d1*z+d2*(torch::pow(z,2)-1);

    return torch::exp(om-g*xi+(b-g*p)*torch::log(h)+tau_z-g*del_z+g*torch::log(x));
}

torch::Tensor NGARCH(torch::TensorList params, torch::Tensor& z, const torch::Tensor& u, torch::Tensor& h) {
    if (params.size() != 4) {
        std::cerr << "NGARCH volatility equation requires 4 parameters." << std::endl;
        return torch::tensor({});
    }
    torch::Tensor omega = params[0];
    torch::Tensor alpha = params[1];
    torch::Tensor phi   = params[2];
    torch::Tensor gamma = params[3];

    return omega + phi * (h - omega) + alpha*h*(torch::pow(z,2)+2.0*gamma*z-1.0);
}

torch::Tensor GJR(torch::TensorList params, torch::Tensor& z, const torch::Tensor& u, torch::Tensor& h) {
    if (params.size() != 4) {
        std::cerr << "GJR volatility equation requires 4 parameters." << std::endl;
        return torch::tensor({});
    }
    torch::Tensor omega = params[0];
    torch::Tensor alpha = params[1];
    torch::Tensor phi   = params[2];
    torch::Tensor gamma = params[3];

    // Implement GJR processing
    return omega + phi * (h - omega) + alpha * h * (torch::pow(z, 2) - 1)
           - gamma * h * (torch::pow(torch::maximum(torch::zeros_like(z), -z), 2) - 0.5);
}

void initializeFuncMap() {
    funcMap["EGARCH"] = EGARCH;
    funcMap["GJR"] = GJR;
    funcMap["NGARCH"] = NGARCH;
}

torch::Tensor equation(const std::string& funcName, torch::TensorList params, torch::Tensor& z, const torch::Tensor& u, torch::Tensor& h) {
    initializeFuncMap();  // Ensure the map is initialized before use
    auto it = funcMap.find(funcName);
    if (it != funcMap.end()) {
        return it->second(params, z, u, h);
    } else {
        std::cerr << "Unknown function" << std::endl;
        return torch::tensor({});
    }
}

}  // volatilities

