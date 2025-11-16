#include "VIFunctional.h"
#include "utils.h"
#include <cstdint>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <tuple>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <iomanip>
#include <map>
#include <string>

using namespace volatility_rnn_functional;

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
};

torch::Tensor slope_transform(const torch::Tensor& params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return torch::log((params - lb) / (ub - params)) / slope;
}

torch::Tensor slope_inverse_transform(const torch::Tensor& scaled_params, const torch::Tensor& lb, const torch::Tensor& ub, torch::Tensor slope) {
    return lb + (ub - lb) / (1 + torch::exp(-slope * scaled_params));
}

torch::Tensor fc_layer(const torch::Tensor& rnn_output, 
                            const torch::Tensor& fc_weight, 
                            const torch::Tensor& fc_bias) {
    // Apply FC layer: η_t = RNN_output * W^T + b
    auto eta_t = torch::linear(rnn_output, fc_weight, fc_bias);
    // Apply activation to ensure positive variance
    return torch::softplus(eta_t);
}

std::map<std::string, torch::Tensor> forward_prop(torch::Tensor& data,
                                                    VolatilityRNNParams& params,
                                                    const std::string& model_type,
                                                    const data_utils utils)
{
    auto y = data;
    torch::Tensor rnn_out{}, out_h{}, out_c{};
    torch::Tensor eta_l{torch::empty_like(y)}, z_l{torch::empty_like(y)};
    torch::Tensor h_t = torch::var(y);
    params.initial_variance = h_t;
    torch::Tensor eta_t = params.initial_variance;
    torch::Tensor sc_p = torch::stack({params.garch_model->get_learnable_parameters()});
    torch::Tensor unsc_p = slope_inverse_transform(sc_p, utils.lb, utils.ub, utils.slp);
    torch::Tensor z_t = params.garch_model->generate_shock(eta_t, y[0], unsc_p[3]);

    // Initial hidden states on the correct device
    auto hx = custom::create_initial_hidden(utils.n_layers, false, utils.b_size, utils.hid_size).to(y.device());
    auto cx = torch::zeros_like(hx); // For LSTM

    eta_l.index_put_({0}, eta_t);
    z_l.index_put_({0}, z_t);
    
    for (int64_t t = 1; t < y.size(0); ++t) {
        // Create features tensor with proper dimensions [batch=1, seq=1, features=3]
        auto features = torch::stack({eta_t.squeeze(), y[t-1], z_t}).view({utils.b_size, -1, utils.in_size});
        
        // Create z_sequence with proper dimensions [seq=1, batch=1] for batch_first=true -> [batch=1, seq=1]
        auto z_seq = z_t.unsqueeze(0).unsqueeze(0);  // [1, 1]
        
        if (model_type == "lstm") {
            std::tie(rnn_out, out_h, out_c) = 
                simple::volatility_lstm(features, hx, cx, params, 
                                    true, utils.n_layers, utils.dropout, true, false, true);  // batch_first = true
            
        } else {
            std::tie(rnn_out, out_h) = 
                simple::volatility_gru(features, hx, params, 
                                   true, utils.n_layers, utils.dropout, true, false, true);  // batch_first = true
        }

        // Extract the output from the RNN (should be [batch=1, seq=1, hidden_size])
        auto rnn_output_squeezed = rnn_out.squeeze(1);  // Remove seq dimension -> [batch=1, hidden_size]
        eta_t = fc_layer(rnn_output_squeezed, params.fc_weight, params.fc_bias);
        
        h_t = params.garch_model->update_variance(h_t, z_t, unsc_p);
        z_t = params.garch_model->generate_shock(eta_t.squeeze(), y[t], unsc_p[3]);

        hx = out_h;
        cx = out_c;
        
        params.initial_variance = h_t;
        eta_l.index_put_({t}, eta_t.squeeze());
        z_l.index_put_({t}, z_t);
    }

    std::map<std::string, torch::Tensor> results = {
        {"zl", z_l},
        {"etal", eta_l},
        {"cell", model_type == "lstm" ? out_c : torch::zeros_like(out_h)},
        {"hidden", out_h}
    };

    return results;
}

// ============================================================================
// TRAINING FUNCTION
// ============================================================================

/**
 * @brief Train volatility-integrated model using MLE with unified GARCH interface
 */
std::tuple<std::map<std::string, std::vector<double>>, VolatilityRNNParams> train_volatility_model(
    const std::string& model_type, // "gru" or "lstm"
    std::shared_ptr<GARCHModel> garch_model, // Unified GARCH model interface
    torch::Tensor& returns_data,
    const data_utils p,
    int num_epochs = 1000,  // Reduced for testing
    double learning_rate = 0.001) {
    
    // Create volatility-integrated RNN parameters using the provided GARCH model
    int gate_multiplier = (model_type == "lstm") ? 4 : 3;
    
    // Get device from returns_data
    auto device = returns_data.device();
    
    // Create RNN parameters on the correct device
    auto rnn_params = custom::create_rnn_params_vector(
        p.in_size, p.hid_size, p.n_layers, false, 
        gate_multiplier, true, torch::kFloat32);
    
    // Move RNN parameters to the correct device
    for (auto& param_set : rnn_params) {
        param_set.weight_ih = param_set.weight_ih.to(device);
        param_set.weight_hh = param_set.weight_hh.to(device);
        if (param_set.has_bias()) {
            param_set.bias_ih = param_set.bias_ih.to(device);
            param_set.bias_hh = param_set.bias_hh.to(device);
        }
    }
    
    // Initialize volatility scaling parameter and FC layer on correct device
    torch::TensorOptions options = torch::TensorOptions(torch::kFloat32).device(device);
    auto gamma_f = torch::tensor(1.0, options);
    auto initial_variance = torch::var(returns_data);
    
    double std_val = 1.0 / std::sqrt(p.hid_size);
    auto fc_weight = torch::empty({1, p.hid_size}, options);
    auto fc_bias = torch::empty({1}, options);
    torch::nn::init::uniform_(fc_weight, -std_val, std_val);
    torch::nn::init::uniform_(fc_bias, -std_val, std_val);
    
    // Create volatility RNN parameters
    VolatilityRNNParams vol_params(rnn_params, garch_model, gamma_f, initial_variance, fc_weight, fc_bias);
    
    // Set all parameters as learnable using unified interface
    set_all_parameters_learnable(vol_params, true);
    
    // Collect all parameters for optimizer using unified interface
    auto parameters = collect_all_parameters(vol_params);
    
    // Create optimizer
    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(learning_rate));
    
    std::vector<double> losses, log_likelihoods;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();
        
        // Forward pass
        auto result = forward_prop(returns_data, vol_params, model_type, p);
        auto loss = neg_log_likelihood(result.at("zl"), result.at("etal"));
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping for stability
        torch::nn::utils::clip_grad_norm_(parameters, 1.0);
        
        optimizer.step();
        
        // Store metrics
        losses.push_back(loss.item<double>());
        log_likelihoods.push_back(-loss.item<double>()); // Convert to positive log-likelihood
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << std::fixed << std::setprecision(6) 
                     << loss.item<double>() << ", Model: " << garch_model->get_model_name() << std::endl;
        }
    }
    
    // Return results with trained model parameters
    std::map<std::string, std::vector<double>> results = {
        {"losses", losses},
        {"log_likelihoods", log_likelihoods}
    };
    
    return std::make_tuple(results, vol_params);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

void save_to_csv(const std::string& filename, const std::vector<std::string>& headers, 
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
// MAIN FUNCTION
// ============================================================================

int main() {
    
    auto mps_available = torch::mps::is_available();
    torch::Device device(mps_available ? torch::kMPS : torch::kCPU);
    std::cout << (mps_available ? "MPS available. Training on GPU." : "Training on CPU.") << '\n';

    // Read CSV file
    auto returns_df = DataFrame::load("csv", "../all_data/exreturns.csv");
    auto options_df = DataFrame::load("csv", "../all_data/options.csv");
    auto noise = DataFrame::read_matrix("../all_data/sample_1.csv").to(device);
    
    // filter returns
    torch::Tensor ret_condition = (returns_df->get_col("Date") >= 19990101 & returns_df->get_col("Date") <= 20181231);
    returns_df = returns_df->loc(ret_condition)->to(device);

    // Filter Options
    torch::Tensor op_condition = (options_df->get_col("Date") >= 20140101 & options_df->get_col("Date") <= 20181231);
    options_df = options_df->loc(op_condition)->to(device);

    auto returns = returns_df->get_col("exret");

    // Set random seed for reproducibility
    torch::manual_seed(42);

    // Create synthetic data for testing
    // std::cout << "Creating synthetic data..." << std::endl;
    // auto [input_features, returns_data] = create_synthetic_data(1, 1000, 5);
    
    data_utils utils;
    utils.in_size = 3;
    utils.hid_size = 1;
    utils.out_size = 1;
    utils.b_size = 1;
    utils.n_layers = 1;
    utils.dropout = 0.0;

    auto p = torch::tensor({ 1.1587e-4, 4.7111e-6, 0.9628, 2.4338, 186.0823}).to(device);
    torch::Tensor lb = torch::tensor({0.0, 0.0, 0.0, 0.0, 100.0}).to(device);
    torch::Tensor ub = torch::tensor({1.0, 1.0, 1.0, 5.0, 300.0}).to(device);
    torch::Tensor slope = torch::tensor(2.0).to(device);
    torch::Tensor scaled_p = slope_transform(p, lb, ub, slope);

    utils.lb = lb; utils.ub = ub; utils.slp = slope;

    // Model parameters
    std::map<std::string, double> hn_params = {
        {"omega_bar", scaled_p[0].item<double>()}, {"alpha", scaled_p[1].item<double>()}, 
        {"phi", scaled_p[2].item<double>()}, {"lamda", scaled_p[3].item<double>()}, 
        {"gamma", scaled_p[4].item<double>()}
    };
    
    // Create GARCH models using factory pattern - use device-aware options
    torch::TensorOptions tensor_options = torch::TensorOptions(torch::kFloat32).device(device);
    auto hn_model = create_garch_model("hn", hn_params, tensor_options);
    
    // Test different model configurations using unified interface
    std::vector<std::tuple<std::string, std::string, std::shared_ptr<GARCHModel>>> models_to_test = {
        {"LSTM-HN", "lstm", hn_model},
        {"GRU-HN", "gru", hn_model}
    };
    
    std::map<std::string, std::map<std::string, std::vector<double>>> all_results;
    std::map<std::string, VolatilityRNNParams> trained_models;
    std::map<std::string, double> final_log_likelihoods;
    
    for (auto& [model_name, model_type, garch_model] : models_to_test) {
        std::cout << "\nTraining " << model_name << " model..." << std::endl;
        
        auto [training_results, trained_model] = train_volatility_model(
            model_type, garch_model, returns, utils);
        
        all_results[model_name] = training_results;
        trained_models[model_name] = trained_model;
        final_log_likelihoods[model_name] = training_results["log_likelihoods"].back();
        
        std::cout << "Final log-likelihood for " << model_name << ": " 
                 << std::fixed << std::setprecision(6) << final_log_likelihoods[model_name] << std::endl;
        
        // Save each trained model
        save_model_package(trained_model, model_type, training_results, model_name);
    }
    
    // Save training results to CSV
    std::vector<std::string> headers = {"epoch"};
    std::vector<std::vector<double>> csv_data;
    
    // Add epoch numbers
    std::vector<double> epochs;
    for (size_t i = 0; i < all_results.begin()->second["losses"].size(); ++i) {
        epochs.push_back(static_cast<double>(i));
    }
    csv_data.push_back(epochs);
    
    // Add loss and log-likelihood data for each model
    for (auto& [model_name, results] : all_results) {
        headers.push_back(model_name + "_loss");
        headers.push_back(model_name + "_log_likelihood");
        csv_data.push_back(results["losses"]);
        csv_data.push_back(results["log_likelihoods"]);
    }
    
    save_to_csv("training_results.csv", headers, csv_data);
    
    // Find best model
    auto best_model_it = std::max_element(final_log_likelihoods.begin(), final_log_likelihoods.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "MODEL COMPARISON RESULTS" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    for (auto& [model_name, log_likelihood] : final_log_likelihoods) {
        std::cout << model_name << ": Final Log-Likelihood = " 
                 << std::fixed << std::setprecision(6) << log_likelihood << std::endl;
    }
    
    std::cout << "\nBest model: " << best_model_it->first << std::endl;
    
    // Save comparison results to CSV
    std::vector<std::string> comp_headers = {"model", "final_log_likelihood"};
    std::vector<std::vector<double>> comp_data(2);
    
    int model_idx = 0;
    for (auto& [model_name, log_likelihood] : final_log_likelihoods) {
        comp_data[0].push_back(static_cast<double>(model_idx++));
        comp_data[1].push_back(log_likelihood);
    }
    
    save_to_csv("model_comparison.csv", comp_headers, comp_data);
    
    // Demonstrate model loading and usage
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "MODEL PERSISTENCE DEMONSTRATION" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Save the best model separately
    std::string best_model_name = best_model_it->first;
    auto& best_trained_model = trained_models[best_model_name];
    
    // Extract model type from best model name
    std::string best_model_type = (best_model_name.find("LSTM") != std::string::npos) ? "lstm" : "gru";
    
    // Save best model with special name
    save_model_package(best_trained_model, best_model_type, all_results[best_model_name], "BEST_MODEL");
    
    // Demonstrate loading and using saved models
    try {
        // Load the saved model
        auto [loaded_params, loaded_model_type, loaded_garch_type] = load_volatility_model("BEST_MODEL_model.pt");
        
        std::cout << "✓ Successfully loaded best model!" << std::endl;
        std::cout << "  Model type: " << loaded_model_type << "-" << loaded_garch_type << std::endl;
        
        // Test forward pass with loaded model
        std::cout << "\nTesting forward pass with loaded model..." << std::endl;
        auto test_results = forward_prop(returns, loaded_params, loaded_model_type, utils);
        
        std::cout << "✓ Forward pass successful!" << std::endl;
        std::cout << "  Generated " << test_results.at("zl").size(0) << " time steps of estimates" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error in model persistence demo: " << e.what() << std::endl;
    }
    
    // Performance comparison summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "FINAL PERFORMANCE COMPARISON" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << std::left << std::setw(15) << "Model" 
              << std::setw(20) << "Log-Likelihood" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    for (auto& [model_name, _] : trained_models) {
        auto ll = final_log_likelihoods[model_name];
        
        std::cout << std::left << std::setw(15) << model_name
                  << std::setw(20) << std::fixed << std::setprecision(4) << ll
                  << std::endl;
    }
    
    std::cout << "\nImplementation complete!" << std::endl;
    std::cout << "\nFiles created:" << std::endl;
    std::cout << "==============" << std::endl;
    std::cout << "Training Results:" << std::endl;
    std::cout << "- training_results.csv" << std::endl;
    std::cout << "- model_comparison.csv" << std::endl;
    
    std::cout << "\nTrained Models (for each model type):" << std::endl;
    for (const auto& [model_name, _] : trained_models) {
        std::cout << "- " << model_name << "_model.pt (complete model)" << std::endl;
        std::cout << "- " << model_name << "_training.pt (training history)" << std::endl;
        std::cout << "- " << model_name << "_summary.txt (model summary)" << std::endl;
    }
    
    std::cout << "\nBest Model:" << std::endl;
    std::cout << "- BEST_MODEL_model.pt (best performing model)" << std::endl;
    std::cout << "- BEST_MODEL_training.pt (best model training history)" << std::endl;
    std::cout << "- BEST_MODEL_summary.txt (best model summary)" << std::endl;
    
    std::cout << "\nTo use a saved model in downstream tasks:" << std::endl;
    std::cout << "1. Load model: auto [params, type, garch] = load_volatility_model(\"model_file.pt\");" << std::endl;
    std::cout << "2. Use for forward pass: forward_prop(data, params, type, utils);" << std::endl;
    std::cout << "3. All parameters (RNN + GARCH + FC) are preserved and ready to use!" << std::endl;
    
    return 0;
}
