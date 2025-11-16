#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <fstream>
#include <cassert>
#include "ATen/ops/div.h"
#include "ATen/ops/exp.h"
#include "bsIV.h"
#include "dataframe.h"

using torch::indexing::Slice;
using namespace pluss::table;

// Declare your helper functions
torch::Tensor neg_log_likelihood(const torch::Tensor& z, const torch::Tensor& h);
torch::Tensor norm_pdf(torch::Tensor& x, torch::Tensor mu, torch::Tensor sig);
torch::Tensor penalty (torch::Tensor rate, torch::Tensor lb, torch::Tensor ub, torch::Tensor xp);
torch::Tensor unique_tensors(const torch::Tensor& input);
void print_model_parameters(const torch::nn::Module& model);
void to_csv(const std::vector<float>& loss_vec, const std::string& filename);
void load_trained_model(torch::nn::Module& model, const std::string& path);
void report_nan(const torch::Tensor& tensor);

template <typename ModelType>
torch::Tensor final_loss(ModelType& model, torch::Tensor inputs){
      auto [z, h] = model->forward(inputs);
      auto loss = neg_log_likelihood(z, h);
      std::cout << "Final loss: " << loss << "\n";
      return loss;
}

// template functions
template <typename ModelType, typename CriterionType, typename OptimType>
std::tuple<ModelType, std::vector<float>> train_model(ModelType& model, 
                                                   CriterionType& criterion, 
                                                   OptimType& optimizer, 
                                                   int64_t num_epochs, 
                                                   torch::Tensor inputs)
{

   //torch::Tensor loss{}, pen{};
   float running_loss = 0.0;
   std::vector<float> loss_vec{};

   model->train();

   // Training loop
   for (int64_t epoch{0}; epoch < num_epochs; ++epoch) {

      auto [z, h] = model->forward(inputs);
      auto loss = criterion(z, h) + model->pen_val;

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      auto current_loss = loss.template item<float>();
      loss_vec.push_back(current_loss);
      if (std::abs(running_loss - current_loss) <  1e-3){
         std::cout << "Breaking at epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
         break;
      }
      running_loss = current_loss;
      std::cout << "Done with epoch {}: " << epoch << "; " << "Loss: "<< current_loss << "\n";
   }
   
   return std::make_tuple(model, loss_vec);
}

template <typename ModelType>
std::pair<torch::Tensor, torch::Tensor> IVRMSE(ModelType& model,
                                               std::shared_ptr<DataFrame> returns_df, 
                                               std::shared_ptr<DataFrame> options_df,
                                               torch::Tensor& noise)
{
   auto device = noise.device();
   torch::Tensor dates = options_df->get_col("Date");
   torch::Tensor unique_dates = unique_tensors(dates);
   std::vector<torch::Tensor> option_prices{};
   torch::Tensor model_prices{torch::empty_like(options_df->get_col("OptionPrice"))};
   torch::Tensor limit = torch::tensor(1.5).to(device);
   int64_t roll_count = 0;
   int64_t counter = 0;
   TORCH_CHECK(unique_dates.numel() > 0, "Error: unique_dates is empty!");

   for (int64_t i = 0; i < unique_dates.size(0); ++i){
     torch::Tensor condition = (dates == unique_dates[i]);  // Create boolean mask
     auto cross_section = options_df->loc(condition); 
     TORCH_CHECK(cross_section->get_col("StockPrice").size(0) > 0, "Error: Empty cross_section!");
     TORCH_CHECK(cross_section->get_col("Strike").size(0) == cross_section->get_col("StockPrice").size(0),
           "Error: Strike and StockPrice column mismatch!");      
     
     auto [theta_q, sim_returns] = model->simulate_returns(returns_df, cross_section, unique_dates[i], noise, roll_count);
     ++roll_count;
     int64_t N{cross_section->get_col("StockPrice").size(0)};

     for (int64_t j = 0; j < N; ++j){
       auto row_data = cross_section->iloc(j);  // ✅ This is an unordered_map<string, torch::Tensor>

       if (row_data.at("StockPrice").numel() == 0) {
           std::cerr << "Warning: row_data is empty at j=" << j << "\n";
       }
       
       torch::Tensor S0 = row_data.at("StockPrice");  
       torch::Tensor n = row_data.at("bDTM");
       torch::Tensor is_call = row_data.at("isCall");
       torch::Tensor strike = row_data.at("Strike");
       torch::Tensor r = row_data.at("RiskFree") / 252.0;
       torch::Tensor rd = row_data.at("RD");

       torch::Tensor dQdP = theta_q.slice(1, 0, n.item<int64_t>()).prod(1);
       torch::Tensor S_T = S0 * torch::exp(n * rd + sim_returns.slice(1, 0, n.item<int64_t>()).sum(1));

       torch::Tensor payoffs = is_call * torch::max(S_T - strike, torch::zeros_like(S_T)) +
                               (torch::ones_like(is_call) - is_call) * torch::max(strike - S_T, torch::zeros_like(S_T));
       
       torch::Tensor option_price = torch::exp(-r * n) * torch::mean(payoffs * dQdP);
       
       option_prices.push_back(option_price);
       model_prices.index_put_({counter}, option_price);
       ++counter;
     }

   } 
   std::cout << "Done with pricing\n";
   options_df = options_df->set_col("ModelPrice", model_prices);

   auto prices = options_df->get_col("ModelPrice");
   report_nan(prices);
   auto approx_imp_errors = (options_df->get_col("OptionPrice") - options_df->get_col("ModelPrice"))/options_df->get_col("Vega");
   auto approx_ivrmse = torch::sqrt(torch::mean(torch::pow(approx_imp_errors, 2)));

   // implied volatility
   ImpliedVolatility imp_vol(options_df, limit);
   auto implied_vol = imp_vol.getData();
   options_df = options_df->set_col("ModelImpVol", implied_vol);

   auto errors = options_df->get_col("ImpliedVol") - options_df->get_col("ModelImpVol");
   auto ivrmse = torch::sqrt(torch::mean(torch::pow(errors, 2)));
   
   auto compare_prices = options_df->get_cols({"OptionPrice", "ModelPrice", "ImpliedVol", "ModelImpVol"});
   compare_prices->tail(50);

   return {approx_ivrmse, ivrmse};
}

template <typename ModelType>
std::pair<torch::Tensor, torch::Tensor> rnIVRMSE(ModelType& model,
                                               std::shared_ptr<DataFrame> returns_df, 
                                               std::shared_ptr<DataFrame> options_df,
                                               torch::Tensor& noise)
{
   auto device = noise.device();
   torch::Tensor dates = options_df->get_col("Date");
   torch::Tensor unique_dates = unique_tensors(dates);
   std::vector<torch::Tensor> option_prices{};
   torch::Tensor model_prices{torch::empty_like(options_df->get_col("OptionPrice"))};
   torch::Tensor limit = torch::tensor(1.5).to(device);
   int64_t roll_count = 0;
   int64_t counter = 0;
   TORCH_CHECK(unique_dates.numel() > 0, "Error: unique_dates is empty!");

   for (int64_t i = 0; i < unique_dates.size(0); ++i){
     torch::Tensor condition = (dates == unique_dates[i]);  // Create boolean mask
     auto cross_section = options_df->loc(condition); 
     TORCH_CHECK(cross_section->get_col("StockPrice").size(0) > 0, "Error: Empty cross_section!");
     TORCH_CHECK(cross_section->get_col("Strike").size(0) == cross_section->get_col("StockPrice").size(0),
           "Error: Strike and StockPrice column mismatch!");      
     
     auto sim_returns = model->risk_neutralize(returns_df, cross_section, unique_dates[i], noise, roll_count);
     ++roll_count;
     int64_t N{cross_section->get_col("StockPrice").size(0)};

     for (int64_t j = 0; j < N; ++j){
       auto row_data = cross_section->iloc(j);  
       
       torch::Tensor S0 = row_data.at("StockPrice");  
       torch::Tensor n = row_data.at("bDTM");
       torch::Tensor is_call = row_data.at("isCall");
       torch::Tensor strike = row_data.at("Strike");
       torch::Tensor r = row_data.at("RiskFree") / 252.0;
       torch::Tensor rd = row_data.at("RD");

       torch::Tensor S_T = S0 * torch::exp(n * rd + sim_returns.slice(1, 0, n.item<int64_t>()).sum(1));
       torch::Tensor avg_st = torch::mean(S_T);
       torch::Tensor EMS = S0 * torch::exp(n * rd) * torch::div(S_T, avg_st);  // emprical martingale simulation of Duan & Simonato (1999)

       torch::Tensor payoffs = is_call * torch::max(EMS - strike, torch::zeros_like(EMS)) +
                               (torch::ones_like(is_call) - is_call) * torch::max(strike - EMS, torch::zeros_like(EMS));
       
       torch::Tensor option_price = torch::exp(-r * n) * torch::mean(payoffs);
       
       option_prices.push_back(option_price);
       model_prices.index_put_({counter}, option_price);
       ++counter;
     }

   } 
   std::cout << "Done with pricing\n";
   options_df = options_df->set_col("ModelPrice", model_prices);

   auto prices = options_df->get_col("ModelPrice");
   report_nan(prices);
   auto approx_imp_errors = (options_df->get_col("OptionPrice") - options_df->get_col("ModelPrice"))/options_df->get_col("Vega");
   auto approx_ivrmse = torch::sqrt(torch::mean(torch::pow(approx_imp_errors, 2)));

   // implied volatility
   ImpliedVolatility imp_vol(options_df, limit);
   auto implied_vol = imp_vol.getData();
   options_df = options_df->set_col("ModelImpVol", implied_vol);

   auto errors = options_df->get_col("ImpliedVol") - options_df->get_col("ModelImpVol");
   auto ivrmse = torch::sqrt(torch::mean(torch::pow(errors, 2)));
   
   auto compare_prices = options_df->get_cols({"OptionPrice", "ModelPrice", "ImpliedVol", "ModelImpVol"});
   compare_prices->tail(50);

   return {approx_ivrmse, ivrmse};
}

template <typename ModelType>
void replace_params(ModelType& model, const torch::Tensor& new_params) {
    auto parameters = model->parameters();
    TORCH_CHECK(parameters.size() == new_params.numel(), "Number of provided parameters does not match the number of model parameters.");

    int64_t offset = 0;
    for (auto& param : parameters) {
        auto param_size = param.numel();
        param.set_data(new_params.slice(0, offset, offset + param_size).view(param.sizes()).clone());
        offset += param_size;
    }
}

template <typename ModelType>
void update_params(ModelType& model, const std::vector<torch::Tensor>& new_params) {
    auto parameters = model->parameters();
    TORCH_CHECK(parameters.size() == new_params.size(), "Number of provided tensors does not match the number of model parameters.");

    for (size_t i = 0; i < parameters.size(); ++i) {
        TORCH_CHECK(parameters[i].sizes() == new_params[i].sizes(), "Size mismatch between model parameter and provided tensor at index ", i);
        parameters[i].set_data(new_params[i].clone());
    }
}