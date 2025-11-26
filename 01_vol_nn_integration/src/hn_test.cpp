#include <iostream>
#include <torch/torch.h>
#include "heston_nandi.h"
#include "make_data.h"
#include "utils.h"

// expected interface of ModelType:
struct HNModel {
    double S0;  // spot
    double r;   // risk-free rate
    double q;   // dividend yield

    // characteristic function of log S_T under Q:
    // HN-GARCH params
    double omega_bar;
    double phi;
    double alpha;
    double gamma;
    double lambda;
    std::complex<double> cf(const std::complex<double>& u, double T) const;
};


int main(){

  auto mps_available = torch::mps::is_available();
  torch::Device device(mps_available ? torch::kMPS : torch::kCPU);
  std::cout << (mps_available ? "MPS available. Training on GPU." : "Training on CPU.") << '\n';

  torch::Tensor hn_params = torch::tensor({
        0.000005,   // omega_bar
        0.90,       // phi
        8.7111e-6,       // alpha
        126.9821,       // gamma
        3.4562,       // lambda
        1e-4        // h0
    }, torch::kDouble);

    auto data = simulate_vol_model(
        VolModel::HN,
        hn_params,
        /*S0=*/100.0,
        /*r=*/0.01,
        /*d=*/0.00,
        /*n_paths=*/10000,
        /*n_steps=*/252,
        torch::kCPU
    );

  torch::Tensor options = torch::tensor(
    {{100.0, 0.5},
     {110.0, 1.0},
     {90.0,  2.0}}, torch::kDouble);  // [K, T] rows

  // torch::save(data.R, "../all_data/R.pt");
  // torch::save(data.h, "../all_data/h.pt");
  torch::load(data.R, "../all_data/R.pt");   // load data

  int64_t num_epochs = 10'000;
  std::string vol_type = "HNGARCH";
  std::vector<double> loss_vals{};

  torch::Tensor params = torch::tensor({1.1587e-4, 4.7111e-6, 0.9628, 2.4338, 186.0823}).to(device);      
  torch::Tensor lb = torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0}).to(device);
  torch::Tensor ub = torch::tensor({1.0, 1.0, 1.0, 5.0, 500.0}).to(device);
  torch::Tensor scaler = torch::pow(10, torch::ceil(torch::log10(torch::abs(params))));
  torch::Tensor scaled_params = torch::div(params, scaler);
  lb = torch::div(lb, scaler);
  ub = torch::div(ub, scaler);
  
  // Initialize the Heston and Nandi model
  auto model = HestonNandi(scaled_params, lb, ub, scaler);
  model->to(device);
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
  
  std::tie(model, loss_vals) = train_model(model, neg_log_likelihood, optimizer, num_epochs, {data.R,});

  model->to(torch::kCPU);
  torch::save(model, "../models/model_hn19.pt");

  {
      torch::NoGradGuard no_grad;
      // torch::load(model, "../models/model_hn19.pt");
      std::cout << model->parameters() << "\n";
      model->eval();
     
      // option pricing
      auto call_prices = Pricer(model, options, /*is_call=*/true);
      
      // std::cout << "Prices: " << call_prices << "\n";
    }  
  
  std::cout << "Done! \n";

  return 0;
}