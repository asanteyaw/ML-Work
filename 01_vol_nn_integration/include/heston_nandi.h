#pragma once

#include <torch/torch.h>
#include "GARCH_Volatility.h"

using torch::indexing::Slice;

struct HestonNandiImpl : torch::nn::Module {

   // Constructor declaration
   HestonNandiImpl(torch::Tensor params, torch::Tensor lb, torch::Tensor ub, torch::Tensor sc);

   // Forward method declaration
   std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& data);
   torch::Tensor get_update(torch::Tensor& data, torch::Tensor prev_var, int64_t calls);
   torch::Tensor unscale_parameters(torch::Tensor sc_params);
   void get_unscaled_params();
   std::pair<torch::Tensor, torch::Tensor> penalty (torch::Tensor xp, torch::Tensor rate);
   double generate_random_value(double lower_bound, double upper_bound);

   // Analytical pricing using Heston-Nandi closed-form formula



   torch::Tensor omega{}, alpha{}, phi{}, lamda{}, gamma{};
   torch::Tensor lb{}, ub{}, scaler{}, pen_val{}, a0{}, a1{}, si{}, theta0{};
   torch::Tensor m_date{}, m_var{};
   garch_type_model::HNModel hn_model;
   const std::string vol_type;
};
TORCH_MODULE(HestonNandi);

// Constructor implementation
inline HestonNandiImpl::HestonNandiImpl(torch::Tensor params, torch::Tensor lb, torch::Tensor ub, torch::Tensor sc) 
      :lb(lb), ub(ub), scaler(sc), hn_model(params)
{

    //{-12.0, -1.0, -1.0, 0.0, -5.0}  {0.0, 0.0, 0.0, 0.0, -5.0}
    //{-3.0, 1.0, 1.0, 5.0, 0.0}  {1.0, 1.0, 1.0, 5.0, 0.0}
   omega = register_parameter("omega", params[0]);
   alpha = register_parameter("alpha", params[1]);
   phi = register_parameter("phi", params[2]);
   lamda = register_parameter("lamda", params[3]);
   gamma = register_parameter("gamma", params[4]);

}

inline torch::Tensor HestonNandiImpl::unscale_parameters(torch::Tensor sc_params) {
   
   auto unscaled_params = sc_params * scaler;

   auto rate = (torch::tensor(10).pow(20)).to(unscaled_params.device());
   auto[theta, pen] = penalty(unscaled_params, rate);

   pen_val = pen;
   return unscaled_params;
}

inline std::pair<torch::Tensor, torch::Tensor> HestonNandiImpl::penalty (torch::Tensor xp, torch::Tensor rate)
{
   torch::Tensor lb_s = lb * scaler;
   torch::Tensor ub_s = ub * scaler;
   torch::Tensor xc = torch::min(torch::max(xp, lb_s), ub_s);
   torch::Tensor pen = torch::max(torch::abs(xp - xc));

   return {xc, rate * pen};
}

inline void HestonNandiImpl::get_unscaled_params() { // when parameter is unused, could just place the type

    auto scaled_params = torch::stack({omega,alpha,phi,lamda,gamma});
    auto params = scaled_params * scaler;
    int idx = 0;
    for (auto& p : this->parameters()) {
        p.copy_(params[idx]);
        ++idx;
    }
}

inline double HestonNandiImpl::generate_random_value(double lower_bound, double upper_bound) {
    return lower_bound + (upper_bound - lower_bound) * torch::rand(1).item<double>();
}

// Forward method implementation
inline std::pair<torch::Tensor, torch::Tensor> HestonNandiImpl::forward(const torch::Tensor& data) {
   torch::Tensor h_t{}, z_t{}, tz{};
   auto y = data;   // excess returns

   auto hl{torch::empty_like(y)}, zl{torch::empty_like(y)};
  
   auto scaled_params = torch::stack({omega,alpha,phi,lamda,gamma});
   torch::Tensor params = unscale_parameters(scaled_params);
   
   torch::Tensor om{params[0]}, al{params[1]}, ph{params[2]}, la{params[3]}, ga{params[4]};
   
   hn_model.set_params({om, al, ph, la, ga});
   h_t = torch::var(y);
   z_t = hn_model.generate_shock(h_t, y[0]);

   hl.index_put_({0}, h_t);
   zl.index_put_({0}, z_t);
   
   for (int64_t t = 1; t < y.size(0); ++t) {
      h_t = hn_model.update_variance(h_t, z_t);
      z_t = hn_model.generate_shock(h_t, y[t]);

      hl.index_put_({t}, h_t);
      zl.index_put_({t}, z_t);

   }
   
   return {zl, hl};
}

inline torch::Tensor HestonNandiImpl::get_update(torch::Tensor& data, torch::Tensor prev_var, int64_t calls){

   torch::Tensor h_t{}, u_t{}, z_t{};
   torch::Tensor y = data;

   if (calls == 0) {
      h_t = torch::var(y);
   }else{
      h_t = prev_var;
   }
   z_t = hn_model.generate_shock(h_t, y[0]);

   for (int64_t t = 1; t < y.size(0); ++t) {
      h_t = hn_model.update_variance(h_t, z_t);
      z_t = hn_model.generate_shock(h_t, y[t]);

   }
   return h_t;
}



