// Functionals.h
#pragma once

#include <torch/torch.h>
#include <vector>

namespace volatilities {

std::unordered_map<std::string, std::function<torch::Tensor(torch::TensorList, torch::Tensor&, torch::Tensor&)>> funcMap;

torch::Tensor EGARCH(torch::TensorList params, torch::Tensor& z, torch::Tensor& h) {
    if (params.size() != 4) {
        std::cerr << "EGARCH requires 4 parameters." << std::endl;
        return torch::tensor({});
    }
    return torch::exp(params[0] + params[2] * (torch::log(h) - params[0]) + params[1] * (torch::abs(z) - torch::sqrt(torch::tensor(2.0 / M_PI))) + params[3] * z);
}

torch::Tensor NGARCH(torch::TensorList params, torch::Tensor& z, torch::Tensor& h) {
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

torch::Tensor GJR(torch::TensorList params, torch::Tensor& z, torch::Tensor& h) {
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

torch::Tensor equation(const std::string& funcName, torch::TensorList params, torch::Tensor& z, torch::Tensor& h) {
    initializeFuncMap();  // Ensure the map is initialized before use
    auto it = funcMap.find(funcName);
    if (it != funcMap.end()) {
        return it->second(params, z, h);
    } else {
        std::cerr << "Unknown function" << std::endl;
        return torch::tensor({});
    }
}

}  // volatilities

namespace transforms {

// Linear Transformation
torch::Tensor linear_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return lb + (ub - lb) * params;
}

torch::Tensor linear_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return (scaled_params - lb) / (ub - lb);
}

// Tanh Transformation
torch::Tensor tanh_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return lb + (ub - lb) * (1 + torch::tanh(params)) / 2;
}

torch::Tensor tanh_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return torch::atanh(2 * (scaled_params - lb) / (ub - lb) - 1);
}

// Slope Transformation
torch::Tensor slope_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return torch::log((params - lb) / (ub - params)) / slope;
}

torch::Tensor slope_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return lb + (ub - lb) / (1 + torch::exp(-slope * scaled_params));
}

// Exponential Transformation
torch::Tensor exponential_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return lb + (ub - lb) * torch::exp(params) / (1 + torch::exp(params));
}

torch::Tensor exponential_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return torch::log((scaled_params - lb) / (ub - scaled_params));
}

// Log Transformation
torch::Tensor log_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return lb + (ub - lb) * torch::log1p(params - lb) / torch::log1p(ub - lb);
}

torch::Tensor log_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return torch::exp((scaled_params - lb) * torch::log1p(ub - lb) / (ub - lb)) + lb - 1;
}

// Sigmoid Transformation
torch::Tensor sigmoid_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return lb + (ub - lb) / (1 + torch::exp(-params));
}

torch::Tensor sigmoid_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub) {
    return -torch::log((ub - lb) / (scaled_params - lb) - 1);
}

// Softplus Transformation
torch::Tensor softplus_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor param_max) {
    return lb + (ub - lb) * (torch::log1p(torch::exp(params)) / torch::log1p(torch::exp(param_max)));
}

torch::Tensor softplus_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor param_max) {
    torch::Tensor exponent = ((scaled_params - lb) / (ub - lb)) * torch::log1p(torch::exp(param_max));
    return torch::log(torch::exp(exponent) - 1);
}

}  // transforms
