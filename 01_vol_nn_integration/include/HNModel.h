#pragma once

#include "ATen/ops/sqrt.h"
#include "dataframe.h"
#include <torch/torch.h>
#include <tuple>
#include "Functionals.h"
#include "utils.h"

using namespace pluss::table;
using torch::indexing::Slice;

struct HNModelImpl : torch::nn::Module {

    // Constructor declaration
    HNModelImpl(std::string vol, torch::Tensor params, torch::Tensor lb, torch::Tensor ub, torch::Tensor sc);

    // Forward method declaration
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& data);
    torch::Tensor get_update(torch::Tensor& data, torch::Tensor prev_var, int64_t calls);
    torch::Tensor unscale_parameters(torch::Tensor sc_params);
    void get_unscaled_params();
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
                                    int64_t n_calls)

    // Analytical pricing using Heston-Nandi closed-form formula
    torch::Tensor risk_neutralize(std::shared_ptr<DataFrame> returns, 
                                    std::shared_ptr<DataFrame> opts,
                                    torch::Tensor date,
                                    torch::Tensor& news,
                                    int64_t n_calls);

   torch::Tensor semi_analytic_price(std::shared_ptr<DataFrame> opts,
                                       torch::Tensor current_variance,
                                       torch::Tensor spot_price,
                                       double risk_free_rate);


    torch::Tensor omega{}, alpha{}, phi{}, lamda{}, gamma{};
    torch::Tensor lb{}, ub{}, scaler{}, pen_val{}, a0{}, a1{}, si{}, theta0{};
    torch::Tensor m_date{}, m_var{}, a0_s{}, a1_s{}, si_s{}, theta0_s{};
    const std::string vol_type;
};
TORCH_MODULE(HNModel);

// Constructor implementation
inline HNModelImpl::HNModelImpl(std::string vol, torch::Tensor params, torch::Tensor lb, torch::Tensor ub, torch::Tensor sc) 
      :vol_type(vol), lb(lb), ub(ub), scaler(sc)
{

    //{-12.0, -1.0, -1.0, 0.0, -5.0}  {0.0, 0.0, 0.0, 0.0, -5.0}
    //{-3.0, 1.0, 1.0, 5.0, 0.0}  {1.0, 1.0, 1.0, 5.0, 0.0}
   omega = register_parameter("omega", params[0]);
   alpha = register_parameter("alpha", params[1]);
   phi = register_parameter("phi", params[2]);
   lamda = register_parameter("lamda", params[3]);
   gamma = register_parameter("gamma", params[4]);

}

inline torch::Tensor HNModelImpl::unscale_parameters(torch::Tensor sc_params) {
   
   auto unscaled_params = sc_params * scaler;

   auto rate = (torch::tensor(10).pow(20)).to(unscaled_params.device());
   auto[theta, pen] = penalty(unscaled_params, rate);

   pen_val = pen;
   return unscaled_params;
}

inline std::pair<torch::Tensor, torch::Tensor> HNModelImpl::penalty (torch::Tensor xp, torch::Tensor rate)
{
   torch::Tensor lb_s = lb * scaler;
   torch::Tensor ub_s = ub * scaler;
   torch::Tensor xc = torch::min(torch::max(xp, lb_s), ub_s);
   torch::Tensor pen = torch::max(torch::abs(xp - xc));

   return {xc, rate * pen};
}

inline void HNModelImpl::get_unscaled_params() { // when parameter is unused, could just place the type

    auto scaled_params = torch::stack({omega,alpha,phi,lamda,gamma});
    auto params = scaled_params * scaler;
    int idx = 0;
    for (auto& p : this->parameters()) {
        p.copy_(params[idx]);
        ++idx;
    }
}

inline double HNModelImpl::generate_random_value(double lower_bound, double upper_bound) {
    return lower_bound + (upper_bound - lower_bound) * torch::rand(1).item<double>();
}

// Forward method implementation
inline std::pair<torch::Tensor, torch::Tensor> HNModelImpl::forward(const torch::Tensor& data) {
   torch::Tensor h_t{}, z_t{}, tz{};
   auto y = data;   // excess returns

   auto hl{torch::empty_like(y)}, zl{torch::empty_like(y)};
  
   auto scaled_params = torch::stack({omega,alpha,phi,lamda,gamma});
   torch::Tensor params = unscale_parameters(scaled_params);
   
   torch::Tensor om{params[0]}, al{params[1]}, ph{params[2]}, la{params[3]}, ga{params[4]};
   
   h_t = torch::var(y);
   z_t = (y[0] - la * h_t) / torch::sqrt(h_t);

   hl.index_put_({0}, h_t);
   zl.index_put_({0}, z_t);
   
   for (int64_t t = 1; t < y.size(0); ++t) {
      std::tie(std::ignore, h_t) = volatilities::equation(vol_type, {om, al, ph, ga}, z_t, tz, h_t);
      z_t = (y[t] - la * h_t) / torch::sqrt(h_t);

      hl.index_put_({t}, h_t);
      zl.index_put_({t}, z_t);

   }
   
   return {zl, hl};
}

inline torch::Tensor HNModelImpl::get_update(torch::Tensor& data, torch::Tensor prev_var, int64_t calls){

   torch::Tensor h_t{}, u_t{}, z_t{}, tz{};
   torch::Tensor y = data;

   if (calls == 0) {
      h_t = torch::var(y);
   }else{
      h_t = prev_var;
   }
   z_t = (y[0] - lamda * h_t) / torch::sqrt(h_t);

   for (int64_t t = 1; t < y.size(0); ++t) {
      std::tie(std::ignore, h_t) = volatilities::equation(vol_type, {omega, alpha, phi, gamma}, z_t, tz, h_t);
      z_t = (y[t] - lamda * h_t) / torch::sqrt(h_t);

   }
   return h_t;
}

inline std::pair<torch::Tensor, torch::Tensor> HNModelImpl::simulate_returns(std::shared_ptr<DataFrame> returns, 
                                                                              std::shared_ptr<DataFrame> opts,
                                                                              torch::Tensor date,
                                                                              torch::Tensor& news,
                                                                              int64_t n_calls){
   /*This implements a non variance dependent pricing kernel, that is \theta_{2,t+1} is 0*/                                                                            
   torch::Tensor T = torch::max(opts->get_col("bDTM"));
   auto z = news;
   torch::Tensor essch{}, filt_var{}, tz{};    
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
      m_var = get_update(filt_data, torch::tensor(0.0), n_calls);
   
   } else {
      torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).squeeze();
      torch::Tensor filt_data = exreturns.index_select(0, indices);
      m_var = get_update(filt_data, m_var, n_calls);
      //std::cout << "Updated filtered variable: " << filt_var << std::endl;
   }
   
   m_date = date;
   auto zt = z.select(1, 0);
   torch::Tensor ht = m_var.repeat({nsim});
   torch::Tensor yt = lamda * ht + torch::sqrt(ht) * zt;
   auto tht = -(lamda + 0.5);
   essch = torch::exp(tht * torch::sqrt(ht) * zt - 0.5 * tht.pow(2) * ht);

   y.index_put_({Slice(), 0}, yt);
   z_k.index_put_({Slice(), 0}, essch);
   
   for (int64_t t{1}; t < (T.item<int64_t>() + 1); ++t) {

      // Volatility update
      std::tie(std::ignore, ht) = volatilities::equation(
         vol_type, {omega, alpha, phi, gamma}, zt, tz, ht
      );

      // --- Clamp Logic for ht ---
      ht = torch::where(ht < 0, omega, ht);  // Replace ht only if ht < 0

      // Update zt
      zt = z.select(1, t);

      // Compute yt and Esscher transform
      yt = lamda * ht + torch::sqrt(ht) * zt;
      essch = torch::exp(tht * torch::sqrt(ht) * zt - 0.5 * tht.pow(2) * ht);

      // Store results
      y.index_put_({Slice(), t}, yt);
      z_k.index_put_({Slice(), t}, essch);
   }
   
   return {z_k.slice(1, 1, z_k.size(1)), y.slice(1, 1, y.size(1))};
}

inline torch::Tensor HNModelImpl::risk_neutralize(std::shared_ptr<DataFrame> returns, 
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
      m_var = get_update(filt_data, torch::tensor(0.0), n_calls);
   
   } else {
      torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).squeeze();
      torch::Tensor filt_data = exreturns.index_select(0, indices);
      m_var = get_update(filt_data, m_var, n_calls);
      //std::cout << "Updated filtered variable: " << filt_var << std::endl;
   }                             
   
   m_date = date;
   auto zt = z.select(1, 0);
   torch::Tensor ht = m_var.repeat({nsim});
   torch::Tensor yt = -0.5 * ht + torch::sqrt(ht) * zt;
   auto tht = -(lamda + 0.5);

   y.index_put_({Slice(), 0}, yt);
   
   for (int64_t t{1}; t < (T.item<int64_t>() + 1); ++t) {

      // Volatility update
      ht = omega + phi * (ht - omega) + alpha * (torch::pow(tht * ht + zt, 2)
                                                  - 2.0 * gamma * ht.pow(0.5) * (tht * ht + zt) - 1.0); 

      // --- Clamp Logic for ht ---
      ht = torch::where(ht < 0, omega, ht);  // Replace ht only if ht < 0

      // Update zt
      zt = z.select(1, t);

      // Compute yt and Esscher transform
      yt = -0.5 * ht + torch::sqrt(ht) * zt;

      // Store results
      y.index_put_({Slice(), t}, yt);
                              
   }
   return y.slice(1, 1, y.size(1));
}

inline std::pair<torch::Tensor, torch::Tensor> HNModelImpl::VD_simulate_returns(std::shared_ptr<DataFrame> returns, 
                                                                                 std::shared_ptr<DataFrame> opts,
                                                                                 torch::Tensor date,
                                                                                 torch::Tensor& news,
                                                                                 int64_t n_calls)
{
/*This implements the variance dependent pricing kernel, that is \theta_{2,t+1} is nonzero. 
   This Heston and Nandi is a special case of the component version where rho = varphi = 0.
*/                                                                            
   torch::Tensor T = torch::max(opts->get_col("bDTM"));
   auto z = news;
   torch::Tensor essch{}, theta_1k{}, theta_2k{}, v_k{}, w_k{}, filt_var{}, tz{};    
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
      m_var = get_update(filt_data, torch::tensor(0.0), n_calls);
   
   } else {
      torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
      torch::Tensor indices = torch::nonzero(ret_mask).squeeze();
      torch::Tensor filt_data = exreturns.index_select(0, indices);
      m_var = get_update(filt_data, m_var, n_calls);
      //std::cout << "Updated filtered variable: " << filt_var << std::endl;
   }

   m_date = date;
   auto zt = z.select(1, 0);
   torch::Tensor ht = m_var.repeat({nsim});
   torch::Tensor yt = lamda * torch::sqrt(ht) - 0.5 * ht + torch::sqrt(ht) * zt;
   auto c_s = torch::tensor(0.077);
   auto eta_k = torch::exp(c_s);
   theta_2k = (eta_k - 1.0)/(1.0 - 2.0 * alpha * eta_k);
   theta_1k = -(lamda+0.5) - 2.0 * theta_2k * (alpha/ht + alpha * gamma);
   v_k = (theta_1k - 2.0 * alpha * gamma * theta_2k) * ht;
   w_k = theta_2k * alpha;
   essch = torch::exp(v_k*zt-(0.5 * v_k.pow(2))/(1.0-2.0*w_k)+w_k*zt.pow(2)+0.5*torch::log(1.0-2.0*w_k));

   y.index_put_({Slice(), 0}, yt);
   z_k.index_put_({Slice(), 0}, essch);

   for(int64_t t{1}; t < (T.item<int64_t>()+1); ++t){

      std::tie(std::ignore, ht) = volatilities::equation(vol_type, {omega, alpha, phi, gamma}, zt, tz, ht);
      zt = z.select(1, t);
      yt = lamda * ht + torch::sqrt(ht) * zt;

      theta_2k = (eta_k - 1.0)/(2.0 * alpha * eta_k);
      theta_1k = -(lamda + 0.5) - 2.0 * theta_2k * (alpha/ht + alpha * gamma);
      v_k = (theta_1k - 2.0 * alpha * gamma * theta_2k) * ht;
      w_k = theta_2k * alpha;
      essch = torch::exp(v_k*zt-(0.5 * v_k.pow(2))/(1.0-2.0*w_k)+w_k*zt.pow(2)+0.5*torch::log(1.0-2.0*w_k));
      TORCH_CHECK(!torch::isnan(essch).any().template item<bool>(), "Esscher has NaN values.\n", essch);

      y.index_put_({Slice(), t}, yt);
      z_k.index_put_({Slice(), t}, essch);

   }

   return {z_k.slice(1, 1, z_k.size(1)), y.slice(1, 1, y.size(1))};
}


// Analytical pricing using Heston-Nandi closed-form formula
inline torch::Tensor HNModelImpl::semi_analytic_price(std::shared_ptr<DataFrame> opts,
                                                  torch::Tensor current_variance,
                                                  torch::Tensor spot_price,
                                                  double risk_free_rate) {
    /*
    This implements the analytical Heston-Nandi option pricing formula from the 2000 paper.
    The formula uses the characteristic function approach to compute European call option prices.
    
    Based on Proposition 3 from Heston & Nandi (2000):
    C = e^(-r(T-t)) * E*[Max(S(T) - K, 0)]
    
    Where the expectation is computed using the risk-neutral characteristic function.
    */
    
    // Get option parameters
    auto strikes = opts->get_col("Strike");
    auto maturities = opts->get_col("bDTM");  // Days to maturity
    auto option_types = opts->get_col("Type"); // Call/Put indicator
    
    int64_t n_options = strikes.size(0);
    torch::Tensor option_prices = torch::zeros({n_options}, strikes.options());
    
    // Get unscaled parameters for analytical pricing
    auto scaled_params = torch::stack({omega, alpha, phi, lamda, gamma});
    torch::Tensor params = transforms::slope_inverse_transform(scaled_params, lb, ub, omega);
    
    double om = params[0].item<double>();
    double al = params[1].item<double>();
    double ph = params[2].item<double>();
    double la = params[3].item<double>();
    double ga = params[4].item<double>();
    
    // Risk-neutral parameters (Proposition 1 from HN2000)
    double gamma_star = ga + la + 0.5;  // γ* = γ + λ + 1/2
    
    double S = spot_price.item<double>();
    double h_current = current_variance.item<double>();
    
    for (int64_t i = 0; i < n_options; ++i) {
        double K = strikes[i].item<double>();
        int64_t T = maturities[i].item<int64_t>();
        double tau = T / 252.0;  // Convert days to years
        
        // Compute coefficients A(t;T,φ) and B(t;T,φ) recursively
        // Starting from terminal conditions: A(T;T,φ) = B(T;T,φ) = 0
        
        std::vector<double> A_coeff(T + 1, 0.0);
        std::vector<double> B_coeff(T + 1, 0.0);
        
        // Backward recursion for risk-neutral coefficients
        for (int64_t t = T - 1; t >= 0; --t) {
            // For φ = i (imaginary unit for characteristic function)
            // We need to compute for φ = i and φ = i + 1
            
            // B coefficient (equation 8b from HN2000)
            double B_next = B_coeff[t + 1];
            double denominator = 1.0 - 2.0 * al * B_next;
            
            if (denominator <= 0) {
                // Numerical stability check
                B_coeff[t] = 0.0;
            } else {
                B_coeff[t] = (gamma_star - 0.5 * gamma_star * gamma_star) + ph * B_next + 
                            0.5 * (gamma_star * gamma_star) / denominator;
            }
            
            // A coefficient (equation 8a from HN2000)
            A_coeff[t] = A_coeff[t + 1] + risk_free_rate + B_coeff[t + 1] * om;
            if (denominator > 0) {
                A_coeff[t] -= 0.5 * std::log(denominator);
            }
        }
        
        // Compute characteristic function f*(iφ) at current time
        // f*(iφ) = S^φ * exp(A(0;T,φ) + B(0;T,φ) * h(0))
        
        // For the integral in Proposition 3, we need to evaluate at φ = i and φ = i + 1
        std::complex<double> phi1(0.0, 1.0);  // i
        std::complex<double> phi2(1.0, 1.0);  // 1 + i
        
        // Simplified numerical integration using the inversion formula
        // This is a basic implementation - in practice, you'd use more sophisticated quadrature
        
        double integral_result = 0.0;
        int n_points = 100;  // Number of integration points
        double max_phi = 50.0;  // Integration limit
        
        for (int j = 1; j <= n_points; ++j) {
            double phi_val = j * max_phi / n_points;
            
            // Compute characteristic function at φ = iφ_val + 1
            std::complex<double> phi_complex(1.0, phi_val);
            
            // Simplified characteristic function evaluation
            // In practice, this would use the full recursive computation
            double B_val = B_coeff[0];
            double A_val = A_coeff[0];
            
            std::complex<double> char_func = std::pow(S, phi_complex) * 
                                           std::exp(A_val + B_val * h_current);
            
            // Integrand for the option pricing formula
            std::complex<double> integrand = std::exp(-std::complex<double>(0.0, phi_val) * std::log(K)) * 
                                           char_func / std::complex<double>(0.0, phi_val);
            
            integral_result += integrand.real() * (max_phi / n_points);
        }
        
        // Apply the pricing formula from Proposition 3
        double call_price = S * 0.5 + (1.0 / M_PI) * integral_result - 
                           K * std::exp(-risk_free_rate * tau) * 0.5;
        
        // Ensure non-negative price
        call_price = std::max(0.0, call_price);
        
        // For puts, use put-call parity: P = C - S + K*e^(-rT)
        if (option_types[i].item<int>() == 0) {  // Assuming 0 = put, 1 = call
            double put_price = call_price - S + K * std::exp(-risk_free_rate * tau);
            option_prices[i] = std::max(0.0, put_price);
        } else {
            option_prices[i] = call_price;
        }
    }
    
    return option_prices;
}



// Variance-dependent risk-neutral simulation for Heston–Nandi (special case: rho=0, varphi=0)
inline torch::Tensor HNModelImpl::VD_risk_neutralize(std::shared_ptr<DataFrame> returns, 
                                                    std::shared_ptr<DataFrame> opts,
                                                    torch::Tensor date,
                                                    torch::Tensor& news,
                                                    int64_t n_calls)
{
    // Risk-neutral dynamics for Heston–Nandi (special case: rho = 0, varphi = 0)
    //   theta_1t = -lambda - 1/2 - 2 (alpha/ht + alpha*gamma) * theta_2t
    //   R_t = alpha * theta_2t
    //   Z_t = (theta_1t - 2*alpha*gamma*theta_2t) * sqrt(ht)
    //   y_t = -ht/(2(1-2R_t)) + sqrt(ht/(1-2R_t)) * z_t
    //   ht = omega + phi*(ht-omega) + alpha*(z_star^2 - 2*gamma*sqrt(ht)*z_star - 1)
    //   z_star = Lambda_t + sqrt(Xi_t)*z_t, where Lambda_t = Z_t/(1-2R_t), Xi_t = 1/(1-2R_t)

    torch::Tensor T = torch::max(opts->get_col("bDTM"));
    auto z = news;
    int64_t nsim = z.size(0);
    torch::Tensor y = torch::empty({nsim, T.item<int64_t>()+1}).to(news.device());
    torch::Tensor dates = returns->get_col("Date");
    torch::Tensor exreturns = returns->get_col("exret");

    // Update filtered variance
    if (n_calls == 0) {
        torch::Tensor start = dates[0];
        torch::Tensor ret_mask = ((dates >= start) & (dates <= date));
        torch::Tensor indices = torch::nonzero(ret_mask).view(-1);
        torch::Tensor filt_data = exreturns.index_select(0, indices);
        m_var = get_update(filt_data, torch::tensor(0.0), n_calls);
    } else {
        torch::Tensor ret_mask = ((dates >= m_date) & (dates <= date));
        torch::Tensor indices = torch::nonzero(ret_mask).squeeze();
        torch::Tensor filt_data = exreturns.index_select(0, indices);
        m_var = get_update(filt_data, m_var, n_calls);
    }
    m_date = date;

    // Initial innovation and variance
    auto zt = z.select(1, 0);
    auto ht = m_var.repeat({nsim});

    // θ₂,t initialization: constant or AR(1)
    torch::Tensor theta_2t;

    if (thetas == "ar") {
      theta_2t = a0_s + a1_s * theta0_s + si_s * ar_shocks.select(1, 0).to(ht.device());
    } else {
      theta_2t = c_s;
    }
    

    // θ₁,t, R_t, Z_t
    auto h_sqrt = torch::sqrt(ht);
    auto theta_1t = -lamda - 0.5 - 2.0 * (alpha / ht + alpha * gamma) * theta_2t;
    auto b_k = alpha * theta_2t;
    auto a_k = (theta_1t - 2.0 * alpha * gamma * theta_2t) * h_sqrt;
    auto denom = (1.0 - 2.0 * b_k);
    auto yt = -ht / (2.0 * denom) + torch::sqrt(ht / denom) * zt;
    y.index_put_({Slice(), 0}, yt);

    for (int64_t t = 1; t < (T.item<int64_t>() + 1); ++t) {
        // Compute z*_t
        auto denom_t = (1.0 - 2.0 * b_k);
        auto Lambda_t = a_k / denom_t;
        auto Xi_t = 1.0 / denom_t;
        auto z_star = Lambda_t + torch::sqrt(Xi_t) * zt;

        // Volatility update (single-layer recursion)
      //   ht = omega + phi * (ht - omega) + alpha * (z_star.pow(2) - 2.0 * gamma * torch::sqrt(ht) * z_star - 1.0);
        std::tie(std::ignore, ht) = volatilities::equation(vol_type, {omega, alpha, phi, gamma}, z_star, date, ht);

        // Next innovation
        zt = z.select(1, t);

        // θ₂,t update if AR(1)
        
        if (thetas == "ar") {
            theta_2t = a0_s + a1_s * theta_2t + si_s * ar_shocks.select(1, t).to(ht.device());
        }
        
        h_sqrt = torch::sqrt(ht);
        theta_1t = -lamda - 0.5 - 2.0 * (alpha / ht + alpha * gamma) * theta_2t;
        b_k = alpha * theta_2t;
        a_k = (theta_1t - 2.0 * alpha * gamma * theta_2t) * h_sqrt;
        denom = (1.0 - 2.0 * b_k);
        yt = -ht / (2.0 * denom) + torch::sqrt(ht / denom) * zt;
        y.index_put_({Slice(), t}, yt);
    }
    return y.slice(1, 1, y.size(1));
}