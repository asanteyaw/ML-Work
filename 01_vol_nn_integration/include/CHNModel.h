#pragma once

#include <ATen/core/ATen_fwd.h>
#include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include "Functionals.h"
#include "dataframe.h"
#include "utils.h"

using namespace pluss::table;
using torch::indexing::Slice;

struct CHNModelImpl : torch::nn::Module {

    // Constructor declaration
   CHNModelImpl(std::string vol, torch::Tensor params, torch::Tensor lb, torch::Tensor ub, torch::Tensor sc);

   // Forward method declaration
   std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& data);
   std::pair<torch::Tensor, torch::Tensor> get_update(torch::Tensor data, torch::TensorList var_list, int64_t calls);
   torch::Tensor unscale_parameters(torch::Tensor sc_params);
   void get_unscaled_params();
   void use_params();
   std::pair<torch::Tensor, torch::Tensor> penalty (torch::Tensor xp, torch::Tensor rate);
   double generate_random_value(double lower_bound, double upper_bound);
   std::pair<torch::Tensor, torch::Tensor> simulate_returns(std::shared_ptr<DataFrame> returns, 
                                                            std::shared_ptr<DataFrame> opts,
                                                            torch::Tensor date,
                                                            torch::Tensor& news,
                                                            int64_t n_calls);
   std::pair<torch::Tensor, torch::Tensor> VD_simulate_returns(std::shared_ptr<DataFrame> returns, 
                                                               std::shared_ptr<DataFrame> opts,
                                                               torch::Tensor date,
                                                               torch::Tensor& news,
                                                               int64_t n_calls);
   
   torch::Tensor VD_risk_neutralize(std::shared_ptr<DataFrame> returns, 
                                       std::shared_ptr<DataFrame> opts,
                                       torch::Tensor date,
                                       torch::Tensor& news,
                                       int64_t n_calls);                                                               
                                                      
   torch::Tensor risk_neutralize(std::shared_ptr<DataFrame> returns, 
                                    std::shared_ptr<DataFrame> opts,
                                    torch::Tensor date,
                                    torch::Tensor& news,
                                    int64_t n_calls);

   torch::Tensor omega{}, alpha{}, phi{}, lamda{}, gamma1{}, gamma2{}, vphi{}, rho{};
   torch::Tensor lb{}, ub{}, scaler{}, pen_val{}, a0{}, a1{}, si{}, theta0{};
   torch::Tensor m_date{}, m_var{}, m_long_var{}, a0_s{}, a1_s{}, si_s{}, theta0_s{};
   const std::string vol_type;
};
TORCH_MODULE(CHNModel);

// Constructor implementation
inline CHNModelImpl::CHNModelImpl(std::string vol, torch::Tensor params, torch::Tensor lb, torch::Tensor ub, torch::Tensor sc) 
      :vol_type(vol), lb(lb), ub(ub), scaler(sc)
{

    //{-12.0, -1.0, -1.0, 0.0, -5.0}  {0.0, 0.0, 0.0, 0.0, -5.0}
    //{-3.0, 1.0, 1.0, 5.0, 0.0}  {1.0, 1.0, 1.0, 5.0, 0.0}
   omega = register_parameter("omega", params[0]);
   alpha = register_parameter("alpha", params[1]);
   phi = register_parameter("phi", params[2]);
   lamda = register_parameter("lamda", params[3]);
   gamma1 = register_parameter("gamma1", params[4]);
   gamma2 = register_parameter("gamma2", params[5]);
   vphi = register_parameter("vphi", params[6]);
   rho = register_parameter("rho", params[7]);

}

inline torch::Tensor CHNModelImpl::unscale_parameters(torch::Tensor sc_params) {
   
   auto unscaled_params = sc_params * scaler;

   auto rate = (torch::tensor(10).pow(20)).to(unscaled_params.device());
   auto[theta, pen] = penalty(unscaled_params, rate);

   pen_val = pen;
   return unscaled_params;
}

inline void CHNModelImpl::use_params(){
   if (thetas == "ar"){
      auto inv_params = inv_transform_params(torch::stack({a0, a1, si, theta0}));
      torch::Tensor rate = (torch::tensor(10).pow(20)).to(inv_params.device());
      auto [theta_uns, pen] = penalty(inv_params, rate);
      a0_s = theta_uns[0];
      a1_s = theta_uns[1];
      si_s = theta_uns[2];
      theta0_s = theta_uns[3];
      pen_val = pen;
   } else{
      auto inv_params = inv_transform_params(c);
      torch::Tensor rate = (torch::tensor(10).pow(20)).to(inv_params.device());
      std::tie(c_s, pen_val) = penalty(inv_params, rate);
   }
}

inline std::pair<torch::Tensor, torch::Tensor> CHNModelImpl::penalty (torch::Tensor xp, torch::Tensor rate)
{
   torch::Tensor lb_s = lb * scaler;
   torch::Tensor ub_s = ub * scaler;
   torch::Tensor xc = torch::min(torch::max(xp, lb_s), ub_s);
   torch::Tensor pen = torch::max(torch::abs(xp - xc));

   return {xc, rate * pen};
}

inline void CHNModelImpl::get_unscaled_params() { // when parameter is unused, could just place the type

    auto scaled_params = torch::stack({omega, alpha, phi, lamda,
                                              gamma1, gamma2, vphi, rho});
    auto params = scaled_params * scaler;
    int idx = 0;
    for (auto& p : this->parameters()) {
        p.copy_(params[idx]);
        ++idx;
    }
}

inline double CHNModelImpl::generate_random_value(double lower_bound, double upper_bound) {
    return lower_bound + (upper_bound - lower_bound) * torch::rand(1).item<double>();
}

// Forward method implementation
inline std::pair<torch::Tensor, torch::Tensor> CHNModelImpl::forward(const torch::Tensor& data) {
   torch::Tensor h_t{}, q_t{}, z_t{};
   auto y = data;   // excess returns
   auto hl{torch::empty_like(y)}, ql{torch::empty_like(y)}, zl{torch::empty_like(y)};
  
   auto scaled_params = torch::stack({omega, alpha, phi, lamda,
                                              gamma1, gamma2, vphi, rho});
   torch::Tensor params = unscale_parameters(scaled_params);
   
   torch::Tensor om{params[0]},al{params[1]},ph{params[2]},la{params[3]},g1{params[4]},g2{params[5]},vp{params[6]},rh{params[7]};
   
   h_t = torch::var(y);
   q_t = h_t;
   z_t = (y[0] - la * h_t) / torch::sqrt(h_t);

   hl.index_put_({0}, h_t);
   ql.index_put_({0}, q_t);
   zl.index_put_({0}, z_t);
   
   for (int64_t t = 1; t < y.size(0); ++t) {
      std::tie(q_t, h_t) = volatilities::equation(vol_type, {om, al, ph, g1, g2, vp, rh}, z_t, q_t, h_t);
      z_t = z_t = (y[t] - la * h_t) / torch::sqrt(h_t);

      hl.index_put_({t}, h_t);
      ql.index_put_({t}, q_t);
      zl.index_put_({t}, z_t);

   }
   // std::cout << "Done with forward: " << sigma <<"\n";
   return {zl, hl};
}


inline std::pair<torch::Tensor, torch::Tensor> CHNModelImpl::get_update(torch::Tensor data, torch::TensorList vars, int64_t calls){

   torch::Tensor h_t{}, u_t{}, z_t{}, q_t{};
   torch::Tensor y = data;

   if (calls == 0) {
      h_t = torch::var(y);
      q_t = h_t;
   }else{
      q_t = vars[0];
      h_t = vars[1];
   }
   z_t = (y[0] - lamda * h_t) / torch::sqrt(h_t);

   for (int64_t t = 1; t < y.size(0); ++t) {
      std::tie(q_t, h_t) = volatilities::equation(vol_type, {omega, alpha, phi, gamma1, gamma2, vphi, rho}, z_t, q_t, h_t);
      z_t = (y[t] - lamda * h_t) / torch::sqrt(h_t);

   }
   return {q_t, h_t};
}

inline std::pair<torch::Tensor, torch::Tensor> CHNModelImpl::simulate_returns(std::shared_ptr<DataFrame> returns,                    
                                                                              std::shared_ptr<DataFrame> opts,
                                                                              torch::Tensor date,
                                                                              torch::Tensor& news,
                                                                              int64_t n_calls){

   torch::Tensor T = torch::max(opts->get_col("bDTM"));
   auto z = news;
   auto device = z.device();
   
   torch::Tensor essch{}, filt_var{}, long_var{};    
   int64_t nsim = z.size(0);
   torch::Tensor y{torch::empty({nsim, T.item<int64_t>()+1}).to(device)};
   torch::Tensor x{torch::empty_like(y)}, h{torch::empty_like(y)};
   torch::Tensor z_k{torch::empty_like(y)};
   torch::Tensor dates = returns->get_col("Date");
   torch::Tensor exreturns = returns->get_col("exret");
   
   if (n_calls == 0) {
      torch::Tensor start = dates[0];  // First date in the tensor
      torch::Tensor ret_mask = ((dates >= start) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).view(-1);
      torch::Tensor filt_data = exreturns.index_select(0, indices);  // Filter returns based on date indices
      std::tie(m_long_var, m_var) = get_update(filt_data, {torch::tensor(-1.0),}, n_calls);
      
   } else {
      torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).view(-1);
      torch::Tensor filt_data = exreturns.index_select(0, indices);
      std::tie(m_long_var, m_var) = get_update(filt_data, {m_long_var, m_var}, n_calls);
   }
   
   m_date = date;
   // m_var = filt_var[-1];
   auto zt = z.select(1, 0);
   torch::Tensor ht = m_var.repeat({nsim});
   torch::Tensor qt = m_long_var.repeat({nsim});
   torch::Tensor yt = lamda * ht + torch::sqrt(ht) * zt;
   auto tht = -(lamda + 0.5);
   essch = torch::exp(tht * torch::sqrt(ht) * zt - 0.5 * tht.pow(2) * ht);

   y.index_put_({Slice(), 0}, yt);
   z_k.index_put_({Slice(), 0}, essch);
   
   for (int64_t t{1}; t < (T.item<int64_t>() + 1); ++t) {
    std::tie(qt, ht) = volatilities::equation(
        vol_type, {omega, alpha, phi, gamma1, gamma2, vphi, rho}, zt, qt, ht
    );

   //  qt = torch::where(qt < 0, omega, qt);  // scalar omega used only where needed
    ht = torch::where(ht < 0, qt, ht);     // use fixed qt if ht is bad

    zt = z.select(1, t);
    yt = lamda * ht + torch::sqrt(ht) * zt;
    essch = torch::exp(tht * torch::sqrt(ht) * zt - 0.5 * tht.pow(2) * ht);

    y.index_put_({Slice(), t}, yt);
    z_k.index_put_({Slice(), t}, essch);
}
   
   return {z_k.slice(1, 1, z_k.size(1)), y.slice(1, 1, y.size(1))};
}

inline torch::Tensor CHNModelImpl::risk_neutralize(std::shared_ptr<DataFrame> returns, 
                                                   std::shared_ptr<DataFrame> opts,
                                                   torch::Tensor date,
                                                   torch::Tensor& news,
                                                   int64_t n_calls)
{
  /*This implements a non variance dependent pricing kernel, that is \theta_{2,t+1} is 0*/                                                                            
   torch::Tensor T = torch::max(opts->get_col("bDTM"));
   auto z = news;
   torch::Tensor filt_var{}, tz{};    
   int64_t nsim = z.size(0);
   torch::Tensor y{torch::empty({nsim, T.item<int64_t>()+1}).to(news.device())};
   torch::Tensor z_k{torch::empty_like(y)};
   torch::Tensor dates = returns->get_col("Date");
   torch::Tensor exreturns = returns->get_col("exret");

   if (n_calls == 0) {
      torch::Tensor start = dates[0];  // First date in the tensor
      torch::Tensor ret_mask = ((dates >= start) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).view(-1);
      torch::Tensor filt_data = exreturns.index_select(0, indices);  // Filter returns based on date indices
      std::tie(m_long_var, m_var) = get_update(filt_data, torch::tensor(0.0), n_calls);
   
   } else {
      torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).squeeze();
      torch::Tensor filt_data = exreturns.index_select(0, indices);
      std::tie(m_long_var, m_var) = get_update(filt_data, m_var, n_calls);
      //std::cout << "Updated filtered variable: " << filt_var << std::endl;
   }                             
   
   m_date = date;
   auto zt = z.select(1, 0);
   torch::Tensor ht = m_var.repeat({nsim});
   torch::Tensor qt = m_long_var.repeat({nsim});
   torch::Tensor yt = -0.5 * ht + torch::sqrt(ht) * zt;
   auto tht = -(lamda + 0.5);

   y.index_put_({Slice(), 0}, yt);
   
   for (int64_t t{1}; t < (T.item<int64_t>() + 1); ++t) {

      // Volatility update
      auto qt_1 = qt;
      qt = omega + rho * (qt - omega) + vphi * (torch::pow(tht * ht + zt, 2)
                                                  - 2.0 * gamma2 * ht.pow(0.5) * (tht * ht + zt) - 1.0);

      qt = torch::where(qt < 0, omega, qt);  // scalar omega used only where needed

      ht = qt + phi * (ht - qt_1) + alpha * (torch::pow(tht * ht + zt, 2)
                                                  - 2.0 * gamma1 * ht.pow(0.5) * (tht * ht + zt) - 1.0); 

      // --- Clamp Logic for ht ---
      ht = torch::where(ht < 0, qt, ht);  // Replace ht only if ht < 0

      // Update zt
      zt = z.select(1, t);

      // Compute yt and Esscher transform
      yt = -0.5 * ht + torch::sqrt(ht) * zt;

      // Store results
      y.index_put_({Slice(), t}, yt);
                              
   }
   return y.slice(1, 1, y.size(1));
}

inline torch::Tensor CHNModelImpl::VD_risk_neutralize(std::shared_ptr<DataFrame> returns, 
                                                   std::shared_ptr<DataFrame> opts,
                                                   torch::Tensor date,
                                                   torch::Tensor& news,
                                                   int64_t n_calls)
{
    // Risk-neutral dynamics for Component Heston–Nandi (variance-dependent)
    // Equations (using your notation):
    //  R_t = (alpha + vphi) * theta_2t                        (b_k)
    //  Z_t = [theta_1t - 2(alpha*gamma1 + vphi*gamma2) theta_2t] * sqrt(h_t)   (a_k)
    //  Lambda_t = Z_t / (1 - 2 R_t),   Xi_t = 1 / (1 - 2 R_t)
    //  z*_t = Lambda_t + sqrt(Xi_t) * z_t
    //  y_t = - h_t / (2 (1 - 2 R_t)) + sqrt(h_t / (1 - 2 R_t)) * z_t
    //  q_{t+1} = omega + rho (q_t - omega) + vphi [ z*_t^2 - 2 gamma2 sqrt(h_t) z*_t - 1 ]
    //  h_{t+1} = q_{t+1} + phi (h_t - q_t) + alpha [ z*_t^2 - 2 gamma1 sqrt(h_t) z*_t - 1 ]
    //  theta_1t = (r - d - mu_t)/h_t - 1/2 - 2 [ (alpha + vphi)/h_t + alpha*gamma1 + vphi*gamma2 ] theta_2t
    //  with mu_t = r - d + lambda h_t  => (r - d - mu_t)/h_t = -lambda
    //  => theta_1t = -lambda - 1/2 - 2 [ (alpha + vphi)/h_t + alpha*gamma1 + vphi*gamma2 ] theta_2t

    torch::Tensor T = torch::max(opts->get_col("bDTM"));
    auto z = news;                  // z ~ N(0,1) innovations
    auto device = z.device();
    int64_t nsim = z.size(0);

    // Output: only risk‑neutral returns y
    torch::Tensor y = torch::empty({nsim, T.item<int64_t>() + 1}, device);

    // Pull return history to filter (q_t, h_t)
    torch::Tensor dates = returns->get_col("Date");
    torch::Tensor exreturns = returns->get_col("exret");

    if (n_calls == 0) {
        torch::Tensor start = dates[0];
        auto ret_mask = ((dates >= start) & (dates <= date));
        auto idx = torch::nonzero(ret_mask).view(-1);
        auto filt_data = exreturns.index_select(0, idx);
        std::tie(m_long_var, m_var) = get_update(filt_data, {torch::tensor(-1.0)}, n_calls);
    } else {
        auto ret_mask = ((dates >= m_date) & (dates <= date));
        auto idx = torch::nonzero(ret_mask).view(-1);
        auto filt_data = exreturns.index_select(0, idx);
        std::tie(m_long_var, m_var) = get_update(filt_data, {m_long_var, m_var}, n_calls);
    }
    m_date = date;

    // Initial states
    auto zt = z.select(1, 0);
    auto ht = m_var.repeat({nsim});
    auto qt = m_long_var.repeat({nsim});

    // θ₂,t: constant vs AR(1) (expects the same members you use elsewhere: thetas, a0_s, a1_s, si_s, ar_shocks, c_s)
    torch::Tensor theta_2t;
    if (thetas == "ar") {
        theta_2t = a0_s + a1_s * theta0_s + si_s * ar_shocks.select(1, 0).to(device);
    } else {
        theta_2t = c_s; // constant
    }

    // θ₁,t (using (r-d-μ_t)/h_t = -lambda)
    auto h_sqrt = torch::sqrt(ht);
    auto theta_1t = -lamda - 0.5 - 2.0 * theta_2t * ( (alpha + vphi) / ht + alpha * gamma1 + vphi * gamma2 );

    // a_k = Z_t, b_k = R_t
    auto b_k = (alpha + vphi) * theta_2t;                                    // R_t
    auto a_k = (theta_1t - 2.0 * (alpha * gamma1 + vphi * gamma2) * theta_2t) * h_sqrt; // Z_t

    // Risk‑neutral return at t=0
    auto denom = (1.0 - 2.0 * b_k);
    auto yt = -ht / (2.0 * denom) + torch::sqrt(ht / denom) * zt;
    y.index_put_({Slice(), 0}, yt);

    // Iterate
    for (int64_t t = 1; t < (T.item<int64_t>() + 1); ++t) {
        // Compute z*_t components (based on current h_t)
        auto denom_t = (1.0 - 2.0 * b_k);
        auto Lambda_t = a_k / denom_t;
        auto Xi_t = 1.0 / denom_t;
        auto z_star = Lambda_t + torch::sqrt(Xi_t) * zt;

        // Update q and h using CNH recursions under Q
        auto q_prev = qt;
        std::tie(qt, ht) = volatilities::equation(vol_type, {omega, alpha, phi, gamma1, gamma2, vphi, rho}, z_star, qt, ht);
      //   qt = omega + rho * (qt - omega) + vphi * ( z_star.pow(2) - 2.0 * gamma2 * torch::sqrt(ht) * z_star - 1.0 );

      //   ht = qt + phi * (ht - q_prev) + alpha * ( z_star.pow(2) - 2.0 * gamma1 * torch::sqrt(ht) * z_star - 1.0 );

        // Next innovation
        zt = z.select(1, t);

        // Update θ₂,t if AR
        if (thetas == "ar") {
            theta_2t = a0_s + a1_s * theta_2t + si_s * ar_shocks.select(1, t).to(device);
        }
        h_sqrt = torch::sqrt(ht);
        theta_1t = -lamda - 0.5 - 2.0 * theta_2t * ( (alpha + vphi) / ht + alpha * gamma1 + vphi * gamma2 );
        b_k = (alpha + vphi) * theta_2t;
        a_k = (theta_1t - 2.0 * (alpha * gamma1 + vphi * gamma2) * theta_2t) * h_sqrt;

        // Risk‑neutral return
        denom = (1.0 - 2.0 * b_k);
        yt = -ht / (2.0 * denom) + torch::sqrt(ht / denom) * zt;

        TORCH_CHECK(!torch::isnan(yt).any().template item<bool>(), "Esscher has NaN values at t=0.\n", essch);
        y.index_put_({Slice(), t}, yt);
    }

    return y.slice(1, 1, y.size(1));
}

inline std::pair<torch::Tensor, torch::Tensor> CHNModelImpl::VD_simulate_returns(std::shared_ptr<DataFrame> returns, 
                                                                                 std::shared_ptr<DataFrame> opts,
                                                                                 torch::Tensor date,
                                                                                 torch::Tensor& news,
                                                                                 int64_t n_calls)
{
/*This implements the variance dependent pricing kernel for Component Heston-Nandi model.
  The variance-dependent pricing kernel is given by:
  Λ_{t+1} = exp(V_{t+1}z_{t+1} + W_{t+1}z_{t+1}² - κ(V_{t+1}, W_{t+1}))
  where V_{t+1} = (θ₁,t+1 - 2(αγ₁ + φγ₂)θ₂,t+1)√h_{t+1}
        W_{t+1} = (α + φ)θ₂,t+1
*/   

   torch::Tensor T = torch::max(opts->get_col("bDTM"));
   auto z = news;
   torch::Tensor essch{}, theta_1k{}, theta_2k{}, v_k{}, w_k{}, filt_var{}, long_var{}, tz{};    
   int64_t nsim = z.size(0);
   torch::Tensor y{torch::empty({nsim, T.item<int64_t>()+1}).to(news.device())};
   torch::Tensor z_k{torch::empty_like(y)};
   torch::Tensor dates = returns->get_col("Date");
   torch::Tensor exreturns = returns->get_col("exret");

   if (n_calls == 0) {
      torch::Tensor start = dates[0];  // First date in the tensor
      torch::Tensor ret_mask = ((dates >= start) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).view(-1);
      torch::Tensor filt_data = exreturns.index_select(0, indices);  // Filter returns based on date indices
      std::tie(m_long_var, m_var) = get_update(filt_data, {torch::tensor(-1.0),}, n_calls);
      
   } else {
      torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).view(-1);
      torch::Tensor filt_data = exreturns.index_select(0, indices);
      std::tie(long_var, filt_var) = get_update(filt_data, {m_long_var, m_var}, n_calls);
      m_long_var = long_var;
      m_var = filt_var;
   }
   
   m_date = date;
   auto zt = z.select(1, 0);
   torch::Tensor ht = m_var.repeat({nsim});
   torch::Tensor qt = m_long_var.repeat({nsim});
   
   // Risk-neutral return dynamics for Component HN
   torch::Tensor yt = lamda * ht + torch::sqrt(ht) * zt;
   
   // Variance-dependent pricing kernel parameters
   // These are typically learned by a neural network, but here we use a simplified form
   auto c_s = torch::tensor(0.077);  // Risk premium parameter
   auto eta_k = torch::exp(c_s);
   
   // θ₂ parameter for variance dependence
   theta_2k = (eta_k - 1.0) / (2.0 * (alpha + vphi) * eta_k);
   
   // θ₁ parameter 
   auto t1 = (alpha + vphi) / ht;
   auto t2 = alpha * gamma1 + vphi * gamma2;
   
   theta_1k = -(lamda + 0.5) - 2.0 * theta_2k * (t1 + t2);
   
   // V and W terms for the pricing kernel
   v_k = (theta_1k - 2.0 * (alpha * gamma1 + vphi * gamma2) * theta_2k) * torch::sqrt(ht);
   w_k = (alpha + vphi) * theta_2k;
   
   // Ensure numerical stability for the pricing kernel
   // w_k = torch::clamp(w_k, -0.49, 0.49);  // Ensure 1-2W > 0
   
   // Variance-dependent Esscher transform
   essch = torch::exp(
       v_k * zt - (0.5 * v_k.pow(2)) / (1.0 - 2.0 * w_k) + 
       w_k * zt.pow(2) + 0.5 * torch::log(1.0 - 2.0 * w_k)
   );
   
   TORCH_CHECK(!torch::isnan(essch).any().template item<bool>(), "Esscher has NaN values at t=0.\n", essch);

   y.index_put_({torch::indexing::Slice(), 0}, yt);
   z_k.index_put_({torch::indexing::Slice(), 0}, essch);
   
   for(int64_t t{1}; t < (T.item<int64_t>()+1); ++t){
      
      // Update component variance dynamics
      std::tie(qt, ht) = volatilities::equation(vol_type, {omega, alpha, phi, gamma1, gamma2, vphi, rho}, zt, qt, ht);
      zt = z.select(1, t);
      
      // Risk-neutral returns
      yt = lamda * ht + torch::sqrt(ht) * zt;
      
      theta_2k = (eta_k - 1.0) / (2.0 * (alpha + vphi) * eta_k);
      t1 = (alpha + vphi) / ht;
      t2 = alpha * gamma1 + vphi * gamma2;
      theta_1k = -(lamda + 0.5) - 2.0 * theta_2k * (t1 + t2);

      // Update V and W terms
      v_k = (theta_1k - 2.0 * (alpha * gamma1 + vphi * gamma2) * theta_2k) * torch::sqrt(ht);
      w_k = (alpha + vphi) * theta_2k;
      
      // Ensure numerical stability
      // w_k = torch::clamp(w_k, -0.49, 0.49);
      
      // Compute variance-dependent pricing kernel
      essch = torch::exp(
          v_k * zt - (0.5 * v_k.pow(2)) / (1.0 - 2.0 * w_k) + 
          w_k * zt.pow(2) + 0.5 * torch::log(1.0 - 2.0 * w_k)
      );
      
      TORCH_CHECK(!torch::isnan(essch).any().template item<bool>(), "Esscher has NaN values at t=", t, "\n", essch);

      y.index_put_({torch::indexing::Slice(), t}, yt);
      z_k.index_put_({torch::indexing::Slice(), t}, essch);
   }

   return {z_k.slice(1, 1, z_k.size(1)), y.slice(1, 1, y.size(1))};
}
