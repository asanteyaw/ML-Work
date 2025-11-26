#include <cstdint>
#include <iostream>
#include "vol_integrated_nn.h"
#include "utils.h"
#include <torch/torch.h>
#include <fstream>
#include <tuple>
#include <vector>
#include "make_data.h"
#include <cmath>
#include <random>
#include <iomanip>
#include <map>
#include <string>


// ============================================================================
// Data Structure
struct data_utils {
    int in_size;   // input size
    int hid_size;  // hidden size
    int out_size;   // output size
    int n_layers;   // num layers
    int b_size;     // batch size
    float dropout;  // dropout probability
    torch::Tensor lb;
    torch::Tensor ub;
    torch::Tensor slp;
    
    torch::Tensor slope_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
        return torch::log((params - lb) / (ub - params)) / slope;
    }
};

torch::Tensor slope_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return torch::log((params - lb) / (ub - params)) / slope;
}

// torch::Tensor slope_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
//     return lb + (ub - lb) / (1 + torch::exp(-slope * scaled_params));
// }

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void to_csv(const std::string& filename, const std::vector<std::string>& headers, 
                 const std::vector<std::vector<double>>& data) {
    std::ofstream file(filename);
    
    // Write headers
    for (size_t i = 0; i < headers.size(); ++i) {
        file << headers[i];
        if (i < headers.size() - 1) file << ",";
    }
    file << "\n";
    
    // Write data
    for (size_t row = 0; row < data[0].size(); ++row) {
        for (size_t col = 0; col < data.size(); ++col) {
            file << std::fixed << std::setprecision(6) << data[col][row];
            if (col < data.size() - 1) file << ",";
        }
        file << "\n";
    }
    
    file.close();
}

// synthetic data for testing
std::pair<torch::Tensor, torch::Tensor> create_synthetic_data(
    int n_samples = 500, int seq_len = 100, int input_size = 5) {
    
    auto input_features = torch::randn({n_samples, seq_len, input_size});
    
    std::vector<std::vector<double>> returns_vec(n_samples, std::vector<double>(seq_len));
    double h_t = 0.01;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int t = 0; t < seq_len; ++t) {
            double z_t = dis(gen);
            h_t = 0.01 + 0.8 * h_t + 0.1 * (z_t * z_t);
            double r_t = std::sqrt(h_t) * z_t;
            returns_vec[i][t] = r_t;
        }
    }
    
    auto returns_data = torch::zeros({n_samples, seq_len});
    for (int i = 0; i < n_samples; ++i) {
        for (int t = 0; t < seq_len; ++t) {
            returns_data[i][t] = returns_vec[i][t];
        }
    }
    
    return std::make_pair(input_features, returns_data);
}

// ============================================================================
// MAIN FUNCTION For Volatility-LSTM
// ============================================================================

int main() {
    
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

  
  torch::load(data.R, "../all_data/R.pt");   // load data

    // Set random seed for reproducibility
    torch::manual_seed(42);
    
    data_utils utils;
    utils.in_size = 4;
    utils.hid_size = 1;
    utils.out_size = 1;
    utils.b_size = 1;
    utils.n_layers = 1;
    utils.dropout = 0.0;
    int64_t n_epochs = 500;

    auto p = torch::tensor({ 1.1587e-4, 4.7111e-6, 0.9628, 2.4338, 186.0823}).to(device);
    torch::Tensor lb = torch::tensor({0.0, 0.0, 0.0, 0.0, 100.0}).to(device);
    torch::Tensor ub = torch::tensor({1.0, 1.0, 1.0, 5.0, 300.0}).to(device);
    torch::Tensor slope = torch::tensor(2.0).to(device);
    torch::Tensor gamma_u = torch::tensor(1.0).to(device);
    torch::Tensor scaled_p = slope_transform(p, lb, ub, slope);
    std::vector<torch::Tensor> aux_vec{gamma_u, lb, ub};

    utils.lb = lb; utils.ub = ub; utils.slp = slope;
    auto vi_hn_model = VolRNNModel(scaled_p, aux_vec, utils.in_size, utils.hid_size, utils.n_layers, /*num_gates=*/4);
    vi_hn_model->to(device);
    
    // Test different model configurations using unified interface
    std::vector<std::tuple<std::string, std::string, VolRNNModel>> models_to_test = {
        {"LSTM-HN", "lstm", vi_hn_model},
        {"GRU-HN", "gru", vi_hn_model}
    };
    
    std::map<std::string,  std::vector<double>> all_results;
    torch::optim::Adam optimizer(vi_hn_model->parameters(), torch::optim::AdamOptions(0.001));
    std::map<std::string, VolRNNModel> trained_models;
    std::map<std::string, double> final_log_likes;
    
    for (auto& [model_name, model_type, vi_model] : models_to_test) {
        std::cout << "\nTraining " << model_name << " model..." << std::endl;
        
        auto [trained_model, training_results] = trainNet(
            vi_model, neg_log_likelihood, optimizer, n_epochs, {data.R, data.h});
        
        all_results[model_name] = training_results;
        trained_models[model_name] = trained_model;
        final_log_likes[model_name] = training_results.back();
        
        std::cout << "Final log-likelihood for " << model_name << ": " 
                 << std::fixed << std::setprecision(6) << final_log_likes[model_name] << std::endl;
        
        // Save each trained model
        // save_model_package(trained_model, model_type, training_results, model_name);
    }
    
    // Save training results to CSV
    std::vector<std::string> headers = {"epoch"};
    std::vector<std::vector<double>> csv_data;
    
    // Add epoch numbers
    std::vector<double> epochs;
    for (size_t i = 0; i < all_results.begin()->second.size(); ++i) {
        epochs.push_back(static_cast<double>(i));
    }
    csv_data.push_back(epochs);
    
    // Add loss and log-likelihood data for each model
    for (auto& [model_name, results] : all_results) {
        headers.push_back(model_name + "_loss");
        // headers.push_back(model_name + "_log_likelihood");
        csv_data.push_back(results);
        // csv_data.push_back(results["log_likelihoods"]);
    }
    
    to_csv("training_results.csv", headers, csv_data);
    

    // ========================================================================
    // POST-TRAINING RISK ANALYSIS: VOL FORECASTING, VaR, CVaR, STRESS TESTS
    // ========================================================================
    using namespace torch::indexing;

    // Use last step across all simulated paths as current horizon
    auto terminal_returns = data.R.index({Slice(), -1});   // [paths]
    auto terminal_vars    = data.h.index({Slice(), -1});   // [paths]

    // Simple one-step-ahead volatility forecast: sqrt(E[h_T])
    double vol_forecast = terminal_vars.mean().sqrt().item<double>();
    std::cout << "\nOne-step-ahead implied volatility forecast: "
              << vol_forecast << std::endl;

    // Historical 1-day VaR and CVaR based on terminal return distribution
    const double alpha_95 = 0.95;
    const double alpha_99 = 0.99;

    auto losses = -terminal_returns; // define loss = -return
    auto sorted_losses = std::get<0>(losses.sort()); // ascending

    int64_t n_paths_mc = sorted_losses.size(0);
    int64_t idx_95 = static_cast<int64_t>(std::floor(alpha_95 * n_paths_mc)) - 1;
    int64_t idx_99 = static_cast<int64_t>(std::floor(alpha_99 * n_paths_mc)) - 1;

    idx_95 = std::max<int64_t>(0, std::min<int64_t>(idx_95, n_paths_mc - 1));
    idx_99 = std::max<int64_t>(0, std::min<int64_t>(idx_99, n_paths_mc - 1));

    double var95 = sorted_losses[idx_95].item<double>();
    double var99 = sorted_losses[idx_99].item<double>();

    auto tail_95 = sorted_losses.index({Slice(idx_95, None)});
    auto tail_99 = sorted_losses.index({Slice(idx_99, None)});

    double cvar95 = tail_95.mean().item<double>();
    double cvar99 = tail_99.mean().item<double>();

    std::cout << "1-day VaR 95% (loss):  " << var95  << std::endl;
    std::cout << "1-day CVaR 95% (loss): " << cvar95 << std::endl;
    std::cout << "1-day VaR 99% (loss):  " << var99  << std::endl;
    std::cout << "1-day CVaR 99% (loss): " << cvar99 << std::endl;

    // Simple stress test: scale shocks by a stress factor
    const double stress_factor = 1.5; // e.g. 50% larger shocks
    auto stressed_losses = stress_factor * losses;
    auto sorted_stressed = std::get<0>(stressed_losses.sort());

    double svar95 = sorted_stressed[ idx_95 ].item<double>();
    double svar99 = sorted_stressed[ idx_99 ].item<double>();

    auto s_tail_95 = sorted_stressed.index({Slice(idx_95, None)});
    auto s_tail_99 = sorted_stressed.index({Slice(idx_99, None)});

    double scvar95 = s_tail_95.mean().item<double>();
    double scvar99 = s_tail_99.mean().item<double>();

    std::cout << "\nSTRESS TEST (" << stress_factor
              << "x shocks) -- VaR/CVaR (loss)" << std::endl;
    std::cout << "Stressed 1-day VaR 95%:  " << svar95  << std::endl;
    std::cout << "Stressed 1-day CVaR 95%: " << scvar95 << std::endl;
    std::cout << "Stressed 1-day VaR 99%:  " << svar99  << std::endl;
    std::cout << "Stressed 1-day CVaR 99%: " << scvar99 << std::endl;

    return 0;
}
