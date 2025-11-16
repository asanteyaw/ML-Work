#include <iostream>
#include <torch/torch.h>
#include "dataframe.h"
#include "GARCH_Volatility.h"
#include "utils.h"

using namespace pluss::table;

int main(){

  auto mps_available = torch::mps::is_available();
  torch::Device device(mps_available ? torch::kMPS : torch::kCPU);
  std::cout << (mps_available ? "MPS available. Training on GPU." : "Training on CPU.") << '\n';

  // Read CSV file
  auto returns_df = DataFrame::load("csv", "../all_data/exreturns.csv")->to(device);
  auto options_df = DataFrame::load("csv", "../all_data/options.csv")->to(device);
  auto noise_mat = DataFrame::read_matrix("../all_data/sample_1.csv");
  
  // filter returns
  torch::Tensor ret_condition = (returns_df->get_col("Date") >= 19960101 & returns_df->get_col("Date") <= 20191231);
  returns_df = returns_df->loc(ret_condition);

  // Filter Options
  torch::Tensor op_condition = (options_df->get_col("Date") >= 20150101 & options_df->get_col("Date") <= 20191231);
  options_df = options_df->loc(op_condition);
  op_condition = (options_df->get_col("bDTM") >= 20);
  options_df = options_df->loc(op_condition);

  auto returns = returns_df->get_col("exret");
  
  auto shock = noise_mat.to(device);
  
  int64_t num_epochs = 10'000;
  std::string vol_type = "CHNGARCH";
  std::vector<float> loss_vals{};

  torch::Tensor params = torch::tensor({1.1594e-4, 2.9188e-6, 0.8877, 2.3879, 332.9349, 134.4245, 1.6101e-6, 0.9905}).to(device);      
  torch::Tensor lb = torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}).to(device);
  torch::Tensor ub = torch::tensor({1.0, 1.0, 1.0, 5.0, 500.0, 500.0, 1.0, 1.0}).to(device);
  torch::Tensor scaler = torch::pow(10, torch::ceil(torch::log10(torch::abs(params))));
  torch::Tensor scaled_params = torch::div(params, scaler);
  lb = torch::div(lb, scaler);
  ub = torch::div(ub, scaler);
  
  // Initialize the Component Heston and Nandi model
  auto model = CHNModel(vol_type, scaled_params, lb, ub, scaler);
  model->to(device);
  torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
  
  // std::tie(model, loss_vals) = train_model(model, neg_log_likelihood, optimizer, num_epochs, {returns,});

  // {
  //   torch::NoGradGuard no_grad;
  //   model->eval();
  //   final_loss(model,  returns);  // calculate final likelihood
  //   model->get_unscaled_params();
  //   print_model_parameters(*model);
        
  // }
  // model->to(torch::kCPU);
  // torch::save(model, "../models/model_chn19.pt");

  {
      torch::NoGradGuard no_grad;

      torch::load(model, "../models/model_chn19.pt");
      std::cout << model->parameters() << "\n";
      model->to(device);
      model->eval();
     
      // monte carlo simulation
      auto [approx_ivrmse, ivrmse] = IVRMSE(model, returns_df, options_df, shock);
      std::cout << "Approx IVRMSE: " << approx_ivrmse << "\n";
      std::cout << "IVRMSE: " << ivrmse << "\n";
    }  
  // options_df.to_csv("../result_tables/real_egarch/output_df19.csv");
  std::cout << "Done! \n";

  return 0;
}