#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include "utils.h"
#include "../include/Functionals.h"

using torch::indexing::Slice;

// Encoder-Decoder Model
struct EconometricModelImpl : torch::nn::Module {

    EconometricModelImpl(std::string vol, size_t n_batch);

    void scale_parameters();
    void get_unscaled_params(int);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor& x);
    std::pair<torch::Tensor,torch::Tensor> predict(torch::Tensor& features);
    std::pair<torch::Tensor, torch::Tensor> simulate_ngarch(torch::Tensor params, size_t T = 20'000);
    std::pair<torch::Tensor, torch::Tensor> ngarch_eq(torch::Tensor& params, const torch::Tensor& returns);  
    torch::Tensor inv_transform_params(torch::Tensor& params, torch::Tensor lb_s, torch::Tensor ub_s);
    torch::Tensor loglikelihood(torch::Tensor& z, torch::Tensor& h);


    // Data Members
    torch::Tensor omega{}, alpha{}, phi{}, lamda{}, gamma{}, lb{}, ub{}, slope{};
    int64_t input_size_{}, num_layers_{}, batch_size_{}, hidden_size_{}, output_size_{};
    torch::Tensor r{torch::tensor(0.019)}, d{torch::tensor(0.012)};   // risk-free rate and dividend yeild
    const std::string vol_type;

};
TORCH_MODULE(EconometricModel);

// TCN Class Implementation

// constructor
inline EconometricModelImpl::EconometricModelImpl(std::string vol, size_t n_batch) 
       : vol_type(vol), batch_size_(n_batch),
         lb(torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0})),
         ub(torch::tensor({1.0, 1.0, 1.0, 5.0, 5.0})), slope(torch::tensor(2.0))
{

    // Create parameters
    omega = register_parameter("omega", generate_random_values(0.000325, 0.000524, n_batch));
    alpha = register_parameter("alpha", generate_random_values(0.048, 0.057, n_batch));
    phi = register_parameter("phi", generate_random_values(0.8, 0.84, n_batch));
    lamda = register_parameter("lamda", generate_random_values(0.025, 0.043, n_batch));
    gamma = register_parameter("gamma", generate_random_values(0.1, 0.6, n_batch));
}

inline std::pair<torch::Tensor, torch::Tensor> EconometricModelImpl::simulate_ngarch(torch::Tensor params, size_t T){
    auto device = torch::kCPU;
    // true values
    auto omega_k = params.select(1, 0);
    auto alpha_k = params.select(1, 1);
    auto phi_k = params.select(1, 2);
    auto lamda_k = params.select(1, 3);
    auto gamma_k = params.select(1, 4);

    // compute initial quantities
    torch::Tensor h_t = omega_k;
    torch::Tensor m_t = r - d + lamda_k * torch::sqrt(h_t) - 0.5 * h_t;
    torch::Tensor z_t = torch::randn(params.size(0)).to(device);
    torch::Tensor x_t = m_t + torch::sqrt(h_t) * z_t;

    // store in containers
    std::vector<torch::Tensor> h_l = {h_t};
    std::vector<torch::Tensor> m_l = {m_t};
    std::vector<torch::Tensor> z_l = {z_t};
    std::vector<torch::Tensor> x_l = {x_t};

    // compute rest of quantities
    for(size_t t{1}; t < T; ++t){
       h_t = omega_k + phi_k * (h_l[t-1] - omega_k) + alpha_k*h_l[t-1]*(torch::pow(z_l[t-1],2)-2.0*gamma_k*z_l[t-1]-1.0);
       m_t = r - d + lamda_k * torch::sqrt(h_t) - 0.5 * h_t;
       z_t = torch::randn(params.size(0)).to(device);
       x_t = m_t + torch::sqrt(h_t) * z_t;

       // update containers
       h_l.push_back(h_t);
       m_l.push_back(m_t);
       z_l.push_back(z_t);
       x_l.push_back(x_t);
    }
    return {torch::stack(x_l,1), torch::stack(h_l,1)};
}

inline void EconometricModelImpl::scale_parameters() {
   auto params = this->parameters();
   int idx = 0;
   for (auto& p : params) {
       p.data() = transforms::slope_transform(p, lb[idx], ub[idx], slope);
       idx++;
   }
}

inline void EconometricModelImpl::get_unscaled_params(int) { // when parameter is unused, could just place the type

    auto params = this->parameters();
    int idx = 0;
    for (auto& p : params) {
        p.copy_(inv_transform_params(p, lb[idx], ub[idx]));
        ++idx;
    }
}

inline torch::Tensor EconometricModelImpl::inv_transform_params(torch::Tensor& param, torch::Tensor lb_s, torch::Tensor ub_s) {
        return transforms::slope_inverse_transform(param, lb_s, ub_s, slope);
}

inline std::pair<torch::Tensor,torch::Tensor> EconometricModelImpl::ngarch_eq(torch::Tensor& params, const torch::Tensor& x){
   size_t T = x.size(1);  // x is returns  
   
   auto omega_s = params.select(1, 0);
   auto alpha_s = params.select(1, 1);
   auto phi_s = params.select(1, 2);
   auto lamda_s = params.select(1, 3);
   auto gamma_s = params.select(1, 4);

   auto rd = r - d;
   torch::Tensor h_t = omega_s;
   torch::Tensor z_t = (x.select(1, 0) - rd + 0.5 * h_t - lamda_s * torch::sqrt(h_t)) / torch::sqrt(h_t);
   std::vector<torch::Tensor> h_l = {h_t};
   std::vector<torch::Tensor> z_l = {z_t};

   for(size_t t{1}; t < T; ++t){
      h_t = omega_s + phi_s * (h_l[t-1] - omega_s) + alpha_s*h_l[t-1]*(torch::pow(z_l[t-1],2)-2.0*gamma_s*z_l[t-1]-1.0);
      z_t = (x.select(1, t) - rd + 0.5 * h_t - lamda_s * torch::sqrt(h_t)) / torch::sqrt(h_t);
      h_l.push_back(h_t);
      z_l.push_back(z_t);
   }

   return {torch::stack(z_l,1), torch::stack(h_l,1)};
}

inline std::pair<torch::Tensor,torch::Tensor> EconometricModelImpl::forward(torch::Tensor& x) {
   
   auto om = inv_transform_params(omega, lb[0], ub[0]).unsqueeze(1);
   auto al = inv_transform_params(alpha, lb[1], ub[1]).unsqueeze(1);
   auto ph = inv_transform_params(phi, lb[2], ub[2]).unsqueeze(1);
   auto la = inv_transform_params(lamda, lb[3], ub[3]).unsqueeze(1);
   auto ga = inv_transform_params(gamma, lb[4], ub[4]).unsqueeze(1);

   auto params = torch::cat({om, al, ph, la, ga}, /*dim=*/-1);
   return ngarch_eq(params, x);   // (batch_size,  output_size)
}

inline torch::Tensor EconometricModelImpl::loglikelihood(torch::Tensor& z, torch::Tensor& h){   
    // likelihood
    auto loglike = torch::sum(0.5*torch::log(torch::tensor(2.0*M_PI))+0.5*torch::log(h)+0.5*torch::pow(z, 2));
    return loglike/batch_size_;
}

inline std::pair<torch::Tensor,torch::Tensor> EconometricModelImpl::predict(torch::Tensor& features){

   //torch::Tensor ctx = torch::zeros(features.size(0), output_size);   

   // process batch-by-batch
   // for(size_t t{0}; t < features.size(0); ++t){
   //   auto out = forward(features[t], ctx);
   //   cxt = out;
   // }

   return forward(features);
}


