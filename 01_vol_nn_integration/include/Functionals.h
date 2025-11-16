// Functionals.h
#pragma once

#include <torch/torch.h>

namespace transforms {
// Slope Transformation
torch::Tensor slope_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return torch::log((params - lb) / (ub - params)) / slope;
}

torch::Tensor slope_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return lb + (ub - lb) / (1 + torch::exp(-slope * scaled_params));
}
}  // namespace transforms

namespace volatilities {

std::unordered_map<std::string, 
    std::function<std::pair<torch::Tensor, 
      torch::Tensor>(torch::TensorList, torch::Tensor&, torch::Tensor&, torch::Tensor&)>> funcMap;

std::pair<torch::Tensor, torch::Tensor> CHNGARCH(torch::TensorList params, torch::Tensor& z, 
                                                torch::Tensor& qt_1, torch::Tensor& h) {
    TORCH_CHECK(params.size() == 7, "Component NGARCH volatility equations requires 7 parameters.");

    torch::Tensor omega = params[0];
    torch::Tensor alpha = params[1];
    torch::Tensor phi   = params[2];
    torch::Tensor g1 = params[3];
    torch::Tensor g2 = params[4];
    torch::Tensor vphi = params[5];
    torch::Tensor rho = params[6];

    torch::Tensor qt = omega + rho * (qt_1 - omega) + vphi * (z.pow(2) - 2.0 * h.pow(0.5) * z * g2 - 1.0);
    torch::Tensor ht = qt + phi * (h - qt_1) + alpha * (z.pow(2) - 2.0 * h.pow(0.5) * g1 * z - 1.0);    

    return {qt, ht};
}

std::pair<torch::Tensor, torch::Tensor> HNGARCH(torch::TensorList params, torch::Tensor& z, const torch::Tensor& u, torch::Tensor& h) {
    TORCH_CHECK(params.size() == 4, "HNGARCH volatility equation requires 4 parameters!");

    torch::Tensor omega = params[0];
    torch::Tensor alpha = params[1];
    torch::Tensor phi   = params[2];
    torch::Tensor gamma = params[3];

    auto ht = omega + phi * (h - omega) + alpha * (z.pow(2) - 2.0 * h.pow(0.5) * gamma * z - 1.0);
    auto dumm = torch::tensor(0);
    return  {dumm, ht}; 
}

void initializeFuncMap() {
    funcMap["CHNGARCH"] = CHNGARCH;
    funcMap["HNGARCH"] = HNGARCH;
}

std::pair<torch::Tensor, torch::Tensor> equation(const std::string& funcName, torch::TensorList params, 
                                                    torch::Tensor& z, torch::Tensor& q, torch::Tensor& h) {
    initializeFuncMap();  // Ensure the map is initialized before use
    auto it = funcMap.find(funcName);
    TORCH_CHECK(it != funcMap.end(), "Unknown function");
    
    return it->second(params, z, q, h);
    
}

}  // volatilities

