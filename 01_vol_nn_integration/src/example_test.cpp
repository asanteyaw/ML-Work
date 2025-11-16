#include "dataframe.h"
#include <torch/torch.h>
#include <iostream>

using namespace pluss::table;

int main(){

  auto mps_available = torch::mps::is_available();
  torch::Device device(mps_available ? torch::kMPS : torch::kCPU);
  std::cout << (mps_available ? "MPS available. Training on GPU." : "Training on CPU.") << '\n';

  // Read CSV file
  auto returns_df = DataFrame::load("csv", "../all_data/returns.csv")->to(device);
  auto rv_df = DataFrame::load("csv", "../all_data/RV5.csv")->to(device);
  auto options_df = DataFrame::load("csv", "../all_data/options.csv")->to(device);
  auto noise1_mat = DataFrame::read_matrix("../all_data/sample_1.csv");
  auto noise2_mat = DataFrame::read_matrix("../all_data/sample_2.csv");

  // filter returns
  torch::Tensor ret_condition = (returns_df->get_col("Date") >= 20000101 & returns_df->get_col("Date") <= 20191231);
  returns_df = returns_df->loc(ret_condition);

  // filter rv
  rv_df = rv_df->loc(ret_condition);

  // Filter Options
  torch::Tensor op_condition = (options_df->get_col("Date") >= 20150101 & options_df->get_col("Date") <= 20191231);
  options_df = options_df->loc(op_condition);

  auto returns = returns_df->get_col("exret");
  auto rv5 = rv_df->get_col("rv5");
  
  auto shock1 = noise1_mat.to(device);
  auto shock2 = noise2_mat.to(device);
  
  int64_t num_epochs = 10'000;
  std::string vol_type = "RealEGARCH";
  std::vector<float> loss_vals{};

  torch::Tensor params = torch::tensor({-0.3308, 0.9650, 1.0451, 0.5116, 0.0212, 0.2565, -0.0403, -0.1297, 0.1085, -0.1751, 0.0399, 0.291556, -0.038, 1.836}).to(device);      
  torch::Tensor lb = torch::tensor({-10.0, -10.0, -10.0, -10.0, 0.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, -10.0, 0.0}).to(device);
  torch::Tensor ub = torch::tensor({10.0, 10.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 10.0, 10.0}).to(device);
  torch::Tensor scaled_params = torch::log((params - lb) / (ub - params)) / 2.0;
  
  // Initialize the Realized EGARCH model
  // auto model = GMRealEGARCHModel(vol_type, scaled_params, lb, ub);
  // model->to(device);
  // torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
  
  // std::tie(model, loss_vals) = trainNet(model, real_gm_log_likelihood, optimizer, num_epochs, {returns, rv5});

  // {
  //   torch::NoGradGuard no_grad;
  //   model->eval();
  //   final_loss(model, real_gm_log_likelihood, {returns, rv5});  // calculate final likelihood
  //   model->get_unscaled_params(0);
  //   print_model_parameters(*model);    
  // }
  // model->to(torch::kCPU);
  // torch::save(model, "../models/model_gm_real_egarch19.pt");
  // to_csv(loss_vals, "../losses/loss_gm_real_egarch19.csv");

  // {
  //     torch::NoGradGuard no_grad;

  //     torch::load(model, "../models/model_gm_real_egarch19.pt");
  //     model->eval();
             
  //     // monte carlo simulation
  //     auto [approx_ivrmse, ivrmse] = IVRMSE(model, returns_df, rv_df, options_df, {shock1, shock2});
  //     std::cout << "Approx IVRMSE: " << approx_ivrmse << "\n";
  //     std::cout << "IVRMSE: " << ivrmse << "\n";
  // }  
  // options_df.to_csv("../result_tables/real_egarch/output_df19.csv");
  std::cout << "Done! \n";

  return 0;
}