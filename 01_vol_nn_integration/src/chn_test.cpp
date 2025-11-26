#include <iostream>
#include <torch/torch.h>
#include "make_data.h"
#include "comp_heston_nandi.h"
#include "utils.h"


int main(){

  auto mps_available = torch::mps::is_available();
  torch::Device device(mps_available ? torch::kMPS : torch::kCPU);
  std::cout << (mps_available ? "MPS available. Training on GPU." : "Training on CPU.") << '\n';

  torch::Tensor chn_params = torch::tensor({
    0.000005,   // omega_bar
    0.90,       // phi
    8.9188e-6,    // alpha
    278.47,       // gamma1
    0.98,       // rho
    4.9188e-6,       // varphi
    111.23,       // gamma2
    1.43,       // lambda
    1e-4,       // h0
    1e-4        // q0
    }, torch::kDouble);

    auto data = simulate_vol_model(
        VolModel::CHN,
        chn_params,
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

  // torch::save(data.R, "../all_data/CR.pt");
  // torch::save(data.h, "../all_data/Ch.pt");
  torch::load(data.R, "../all_data/CR.pt");   // load data
 
  int64_t num_epochs = 10'000;
  std::string vol_type = "CHNGARCH";
  std::vector<double> loss_vals{};

  torch::Tensor params = torch::tensor({1.1594e-4, 2.9188e-6, 0.8877, 2.3879, 332.9349, 134.4245, 1.6101e-6, 0.9905}).to(device);      
  torch::Tensor lb = torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}).to(device);
  torch::Tensor ub = torch::tensor({1.0, 1.0, 1.0, 5.0, 500.0, 500.0, 1.0, 1.0}).to(device);
  torch::Tensor scaler = torch::pow(10, torch::ceil(torch::log10(torch::abs(params))));
  torch::Tensor scaled_params = torch::div(params, scaler);
  lb = torch::div(lb, scaler);
  ub = torch::div(ub, scaler);
  
  // Initialize the Component Heston and Nandi model
  auto model = HestonNandiC(scaled_params, lb, ub, scaler);
  model->to(device);
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
  
  std::tie(model, loss_vals) = train_model(model, neg_log_likelihood, optimizer, num_epochs, {data.R,});

  // model->to(torch::kCPU);
  // torch::save(model, "../models/model_chn19.pt");

  {
      torch::NoGradGuard no_grad;

      // torch::load(model, "../models/model_chn19.pt");
      std::cout << model->parameters() << "\n";
      model->to(device);
      model->eval();
     
      // option pricing
      auto call_prices = Pricer(model, options, /*is_call=*/true);
      
      std::cout << "Prices: " << call_prices << "\n";
    }  
  
  std::cout << "Done! \n";

  return 0;
}