#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include "../libtorch_tft/include/tft_model.h"
#include "../libtorch_tft/include/tft_types.h"
#include "Functionals.h"

using torch::indexing::Slice;

// TFT-based NGARCH Model for parameter learning
struct TFTNGARCHModelImpl : torch::nn::Module {

    TFTNGARCHModelImpl(std::string vol,
                       int64_t input_size,
                       int64_t hidden_size,
                       int64_t num_encoder_steps,
                       int64_t output_size,
                       float dropout_rate = 0.1f,
                       int64_t num_heads = 4);

    torch::Tensor forward(torch::Tensor& x);
    std::pair<torch::Tensor, torch::Tensor> simulate_ngarch(torch::Tensor params, size_t T = 20'000);
    torch::Tensor ngarch_eq(const torch::Tensor& params, const torch::Tensor& returns);  
    torch::Tensor predict(torch::Tensor& features);
    torch::Tensor out_transform(const torch::Tensor& params);
    torch::Tensor params_loss(const torch::Tensor& predicted_params, 
                              const torch::Tensor& returns,
                              const torch::Tensor& true_variances);
    

    // Data Members
    tft::TemporalFusionTransformer tft_model{nullptr};
    torch::nn::MSELoss mse_loss{nullptr};
    torch::nn::Linear parameter_projection{nullptr};
    torch::Tensor omega{}, alpha{}, phi{}, lamda{}, gamma{};
    int64_t input_size_{}, hidden_size_{}, output_size_{}, num_encoder_steps_{};
    torch::Tensor r{torch::tensor(0.019)}, d{torch::tensor(0.012)};   // risk-free rate and dividend yield
    std::vector<std::function<torch::Tensor(torch::Tensor&)>> act_transforms{};
    const std::string vol_type;
    tft::TFTConfig config_;

};
TORCH_MODULE(TFTNGARCHModel);

inline TFTNGARCHModelImpl::TFTNGARCHModelImpl(std::string vol,
                                              int64_t input_size,
                                              int64_t hidden_size,
                                              int64_t num_encoder_steps,
                                              int64_t output_size,
                                              float dropout_rate,
                                              int64_t num_heads)
    : vol_type(vol), 
      input_size_(input_size), 
      hidden_size_(hidden_size), 
      output_size_(output_size),
      num_encoder_steps_(num_encoder_steps)
{
    
    // Configure TFT for NGARCH parameter learning
    config_.input_size = input_size;
    config_.output_size = output_size;  // 5 NGARCH parameters
    config_.hidden_layer_size = hidden_size;
    config_.num_encoder_steps = num_encoder_steps;
    config_.total_time_steps = num_encoder_steps + 1;  // No future forecasting steps
    config_.dropout_rate = dropout_rate;
    config_.num_heads = num_heads;
    config_.quantiles = {0.5f};  // Only predict mean values for parameters
    config_.batch_size = 64;  // Will be overridden during training
    
    // Create TFT model
    tft_model = register_module("tft_model", tft::TemporalFusionTransformer(config_));
    
    // Parameter projection layer to ensure correct output dimensions
    parameter_projection = register_module("parameter_projection", 
                                         torch::nn::Linear(config_.hidden_layer_size, output_size));
    
    mse_loss = register_module("mse_loss", torch::nn::MSELoss());

    // Parameter transformation functions (same as original)
    act_transforms = {
       [](const torch::Tensor& x) { double a = 0.000011, b = 0.0075; return torch::relu(x); },
       [](const torch::Tensor& x) { double a = 0.021, b = 0.089; return torch::relu(x); },  
       [](const torch::Tensor& x) { double a = 0.75, b = 0.98; return torch::sigmoid(x); },     // alpha
       [](const torch::Tensor& x) { double a = 0.025, b = 0.065; return torch::softplus(x); },       // phi
       [](const torch::Tensor& x) { double a = 0.000085, b = 0.0075; return torch::relu(x); },   // omega
    };

}

inline std::pair<torch::Tensor, torch::Tensor> TFTNGARCHModelImpl::simulate_ngarch(torch::Tensor params, size_t T){
    auto device = torch::kCPU;
    // true values
    omega = params.select(1, 0);
    alpha = params.select(1, 1);
    phi = params.select(1, 2);
    lamda = params.select(1, 3);
    gamma = params.select(1, 4);

    // compute initial quantities
    torch::Tensor h_t = omega;
    torch::Tensor m_t = r - d + lamda * torch::sqrt(h_t) - 0.5 * h_t;
    torch::Tensor z_t = torch::randn(params.size(0)).to(device);
    torch::Tensor x_t = m_t + torch::sqrt(h_t) * z_t;

    // store in containers
    std::vector<torch::Tensor> h_l = {h_t};
    std::vector<torch::Tensor> m_l = {m_t};
    std::vector<torch::Tensor> z_l = {z_t};
    std::vector<torch::Tensor> x_l = {x_t};

    // compute rest of quantities
    for(size_t t{1}; t < T; ++t){
       h_t = omega + phi * (h_l[t-1] - omega) + alpha*h_l[t-1]*(torch::pow(z_l[t-1],2)-2.0*gamma*z_l[t-1]-1.0);
       m_t = r - d + lamda * torch::sqrt(h_t) - 0.5 * h_t;
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

inline torch::Tensor TFTNGARCHModelImpl::out_transform(const torch::Tensor& params){
   auto om = 0.00089 + (0.00124 - 0.00089) * torch::sigmoid(params.select(1,0)).unsqueeze(1);
   auto al = 0.011 + (0.21 - 0.011) * torch::sigmoid(params.select(1,1)).unsqueeze(1);
   auto ph = 0.79 + (0.99 - 0.79) * torch::sigmoid(params.select(1,2)).unsqueeze(1);
   auto la = 0.0011 + (0.21 - 0.0011) * torch::sigmoid(params.select(1,3)).unsqueeze(1);
   auto ga = 0.51 + (2.61 - 0.51) * torch::sigmoid(params.select(1,4)).unsqueeze(1);
      
   return torch::cat({om, al, ph, la, ga}, /*dim=*/-1);
}

inline torch::Tensor TFTNGARCHModelImpl::ngarch_eq(const torch::Tensor& params, const torch::Tensor& x){
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

   return torch::stack(h_l, 1);
}

inline torch::Tensor TFTNGARCHModelImpl::forward(torch::Tensor& x) 
{
    // x should be [batch_size, seq_len, input_size]
    // TFT expects input tensor directly, not the TFTData structure
    
    // Forward pass through TFT - it returns predictions and attention weights
    auto [tft_predictions, attention_weights] = tft_model->forward(x);
    
    // tft_predictions shape: [batch_size, forecast_steps, output_size * num_quantiles]
    // For parameter prediction, we want [batch_size, output_size]
    
    // Take the last time step prediction and squeeze to get the parameters
    auto raw_params = tft_predictions.select(1, -1);  // Take last forecast step
    
    // If we have quantiles, take the median (0.5 quantile)
    if (raw_params.size(-1) > output_size_) {
        // Assume quantiles are ordered and take the middle one
        int64_t mid_quantile = config_.quantiles.size() / 2;
        raw_params = raw_params.slice(-1, mid_quantile * output_size_, (mid_quantile + 1) * output_size_);
    }
    
    // Project to correct dimensions if needed
    if (raw_params.size(-1) != output_size_) {
        raw_params = parameter_projection(raw_params);
    }
    
    // Apply parameter transformation constraints
    return out_transform(raw_params);
}

inline torch::Tensor TFTNGARCHModelImpl::predict(torch::Tensor& features){
    return forward(features);
}

inline torch::Tensor TFTNGARCHModelImpl::params_loss(const torch::Tensor& predicted_params, 
                                                     const torch::Tensor& returns, 
                                                     const torch::Tensor& true_variances)
{
    auto predicted_variances = (ngarch_eq(predicted_params, returns.squeeze())).unsqueeze(2);
    auto loss = (losses::huber_per_batch(predicted_variances, true_variances)).mean();
    
    return loss;
}
