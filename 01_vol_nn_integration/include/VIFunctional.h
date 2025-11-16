#pragma once

#include <torch/torch.h>
#include "RNN_Functionals.h"
#include <vector>
#include <tuple>
#include <memory>
#include <fstream>

/**
 * @brief Volatility-integrated functional RNN operations
 * 
 * This namespace extends the custom namespace with volatility-integrated
 * versions of GRU and LSTM cells that incorporate GARCH volatility dynamics
 * into the neural network gates as described in the paper.
 * 
 * Key modifications:
 * - GRU: Volatility integrated into update gate (u_t)
 * - LSTM: Volatility integrated into forget gate (f_t)
 */
namespace volatility_rnn_functional {

// ============================================================================
// GARCH MODEL CLASSES
// ============================================================================

/**
 * @brief Base class for GARCH volatility models with unified parameter management
 */
class GARCHModel {
public:
    virtual ~GARCHModel() = default;
    
    // Core GARCH functionality
    virtual torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params) = 0;
    virtual torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params, 
                                          double r = 0.0, double d = 0.0) = 0;
    virtual torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, torch::Tensor params,
                                            double r = 0.0, double d = 0.0) = 0;
    
    // Unified parameter management interface
    virtual std::vector<torch::Tensor> get_learnable_parameters() = 0;
    virtual void set_parameters_learnable(bool learnable = true) = 0;
    virtual std::string get_model_name() const = 0;
    virtual void initialize_from_config(const std::map<std::string, double>& params) = 0;
    
    // Model serialization interface
    virtual std::map<std::string, torch::Tensor> get_state_dict() const = 0;
    virtual void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) = 0;
};

/**
 * @brief Heston-Nandi GARCH(1,1) model
 */
class HNModel : public GARCHModel {
public:
    torch::Tensor omega_bar, alpha, phi, lamda, gamma;
    
    HNModel(double omega_bar_val, double alpha_val, double phi_val, 
                     double lambda_val, double gamma_val,
                     const torch::TensorOptions& options = torch::kFloat32)
        : omega_bar(torch::tensor(omega_bar_val, options))
        , alpha(torch::tensor(alpha_val, options))
        , phi(torch::tensor(phi_val, options))
        , lamda(torch::tensor(lambda_val, options))
        , gamma(torch::tensor(gamma_val, options)) {}
    
    torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params) override {
        // h_{t+1} = ω̄ + φ(h_t - ω̄) + α(z_t² - 1 - 2γ√h_t z_t)
        auto omega_s = params[0];
        auto alpha_s = params[1];
        auto phi_s = params[2];
        auto lamda_s = params[3];
        auto gamma_s = params[4];
        
        return omega_s + phi_s * (h_t - omega_s) + 
               alpha_s * (z_t.pow(2) - 1 - 2 * gamma_s * torch::sqrt(h_t) * z_t);
    }
    
    torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {
        // R_{t+1} = r - d + λh_{t+1} + √h_{t+1} z_{t+1}, where λ=params
        return r - d + params * h_t + torch::sqrt(h_t) * z_t;
    }

    torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {    
        return (y_t - r + d - params * h_t) / torch::sqrt(h_t);
    }
    
    // Unified parameter management interface implementation
    std::vector<torch::Tensor> get_learnable_parameters() override {
        return {omega_bar, alpha, phi, lamda, gamma};
    }
    
    void set_parameters_learnable(bool learnable = true) override {
        omega_bar.set_requires_grad(learnable);
        alpha.set_requires_grad(learnable);
        phi.set_requires_grad(learnable);
        lamda.set_requires_grad(learnable);
        gamma.set_requires_grad(learnable);
    }
    
    std::string get_model_name() const override {
        return "HN";
    }
    
    void initialize_from_config(const std::map<std::string, double>& params) override {
        if (params.count("omega_bar")) omega_bar = torch::tensor(params.at("omega_bar"));
        if (params.count("alpha")) alpha = torch::tensor(params.at("alpha"));
        if (params.count("phi")) phi = torch::tensor(params.at("phi"));
        if (params.count("lamda")) lamda = torch::tensor(params.at("lamda"));
        if (params.count("gamma")) gamma = torch::tensor(params.at("gamma"));
    }
    
    // Model serialization interface implementation
    std::map<std::string, torch::Tensor> get_state_dict() const override {
        return {
            {"omega_bar", omega_bar},
            {"alpha", alpha},
            {"phi", phi},
            {"lamda", lamda},
            {"gamma", gamma}
        };
    }
    
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) override {
        if (state_dict.count("omega_bar")) omega_bar = state_dict.at("omega_bar");
        if (state_dict.count("alpha")) alpha = state_dict.at("alpha");
        if (state_dict.count("phi")) phi = state_dict.at("phi");
        if (state_dict.count("lamda")) lamda = state_dict.at("lamda");
        if (state_dict.count("gamma")) gamma = state_dict.at("gamma");
    }
};

/**
 * @brief Component Heston-Nandi GARCH model
 */
class CHNModel : public GARCHModel {
public:
    torch::Tensor omega_bar, alpha, phi, lamda, gamma1, gamma2, phi_q, rho;
    torch::Tensor q_t; // Long-run component state
    
    CHNModel(double omega_bar_val, double alpha_val, double phi_val, 
                             double lamda_val, double gamma1_val, double gamma2_val,
                             double phi_q_val, double rho_val,
                             const torch::TensorOptions& options = torch::kFloat32)
        : omega_bar(torch::tensor(omega_bar_val, options))
        , alpha(torch::tensor(alpha_val, options))
        , phi(torch::tensor(phi_val, options))
        , lamda(torch::tensor(lamda_val, options))
        , gamma1(torch::tensor(gamma1_val, options))
        , gamma2(torch::tensor(gamma2_val, options))
        , phi_q(torch::tensor(phi_q_val, options))
        , rho(torch::tensor(rho_val, options))
        , q_t(torch::tensor(omega_bar_val, options)) {}
    
    std::pair<torch::Tensor, torch::Tensor> update_variance_components(
        const torch::Tensor& h_t, const torch::Tensor& q_t_in, const torch::Tensor& z_t) {
        
        auto q_next = omega_bar + rho * (q_t_in - omega_bar) + 
                     phi_q * (z_t.pow(2) - 1 - 2 * gamma2 * torch::sqrt(h_t) * z_t);
        
        auto h_next = q_next + phi * (h_t - q_t_in) + 
                alpha * (z_t.pow(2) - 1 - 2 * gamma1 * torch::sqrt(h_t) * z_t);
        
        return std::make_pair(h_next, q_next);
    }
    
    torch::Tensor update_variance(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params) override {
        auto [h_next, q_next] = update_variance_components(h_t, q_t, z_t);
        q_t = q_next; // Update internal state
        return h_next;
    }
    
    torch::Tensor generate_returns(const torch::Tensor& h_t, const torch::Tensor& z_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {
        return r - d + lamda * h_t + torch::sqrt(h_t) * z_t;
    }

    torch::Tensor generate_shock(const torch::Tensor& h_t, const torch::Tensor& y_t, torch::Tensor params, 
                                  double r = 0.0, double d = 0.0) override {    
        return (y_t - r + d - lamda * h_t) / torch::sqrt(h_t);
    }
    
    // Unified parameter management interface implementation
    std::vector<torch::Tensor> get_learnable_parameters() override {
        return {omega_bar, alpha, phi, lamda, gamma1, gamma2, phi_q, rho};
    }
    
    void set_parameters_learnable(bool learnable = true) override {
        omega_bar.set_requires_grad(learnable);
        alpha.set_requires_grad(learnable);
        phi.set_requires_grad(learnable);
        lamda.set_requires_grad(learnable);
        gamma1.set_requires_grad(learnable);
        gamma2.set_requires_grad(learnable);
        phi_q.set_requires_grad(learnable);
        rho.set_requires_grad(learnable);
    }
    
    std::string get_model_name() const override {
        return "CHN";
    }
    
    void initialize_from_config(const std::map<std::string, double>& params) override {
        if (params.count("omega_bar")) omega_bar = torch::tensor(params.at("omega_bar"));
        if (params.count("alpha")) alpha = torch::tensor(params.at("alpha"));
        if (params.count("phi")) phi = torch::tensor(params.at("phi"));
        if (params.count("lamda")) lamda = torch::tensor(params.at("lamda"));
        if (params.count("gamma1")) gamma1 = torch::tensor(params.at("gamma1"));
        if (params.count("gamma2")) gamma2 = torch::tensor(params.at("gamma2"));
        if (params.count("phi_q")) phi_q = torch::tensor(params.at("phi_q"));
        if (params.count("rho")) rho = torch::tensor(params.at("rho"));
    }
    
    // Model serialization interface implementation
    std::map<std::string, torch::Tensor> get_state_dict() const override {
        return {
            {"omega_bar", omega_bar},
            {"alpha", alpha},
            {"phi", phi},
            {"lamda", lamda},
            {"gamma1", gamma1},
            {"gamma2", gamma2},
            {"phi_q", phi_q},
            {"rho", rho},
            {"q_t", q_t}
        };
    }
    
    void load_state_dict(const std::map<std::string, torch::Tensor>& state_dict) override {
        if (state_dict.count("omega_bar")) omega_bar = state_dict.at("omega_bar");
        if (state_dict.count("alpha")) alpha = state_dict.at("alpha");
        if (state_dict.count("phi")) phi = state_dict.at("phi");
        if (state_dict.count("lamda")) lamda = state_dict.at("lamda");
        if (state_dict.count("gamma1")) gamma1 = state_dict.at("gamma1");
        if (state_dict.count("gamma2")) gamma2 = state_dict.at("gamma2");
        if (state_dict.count("phi_q")) phi_q = state_dict.at("phi_q");
        if (state_dict.count("rho")) rho = state_dict.at("rho");
        if (state_dict.count("q_t")) q_t = state_dict.at("q_t");
    }
};

// ============================================================================
// VOLATILITY-INTEGRATED CELL FUNCTIONS
// ============================================================================

/**
 * @brief Volatility-integrated GRU cell with volatility in update gate
 * @param input Input tensor [batch, input_size]
 * @param hidden Hidden state [batch, hidden_size]
 * @param params GRU parameters (weight matrices are 3*hidden_size)
 * @param garch_model GARCH model for volatility dynamics
 * @param h_var Current variance state
 * @param z_t Random shock for GARCH update
 * @param gamma_u Volatility scaling parameter
 * @param pre_compute_input Whether input is pre-computed
 * @return Tuple of (new_hidden, new_variance)
 */
inline torch::Tensor volatility_gru_cell(
    const torch::Tensor& input,
    const torch::Tensor& hidden,
    const custom::RNNParams& params,
    std::shared_ptr<GARCHModel> garch_model,
    const torch::Tensor& h_var,
    // const torch::Tensor& z_t,
    const torch::Tensor& gamma_u,
    bool pre_compute_input = false) {
    
    // Update GARCH variance
    // torch::Tensor h_var_new = garch_model->update_variance(h_var, z_t);
    
    auto chunked_igates = pre_compute_input ? 
        input.chunk(3, 1) : 
        torch::linear(input, params.weight_ih, params.bias_ih).chunk(3, 1);
    auto chunked_hgates = torch::linear(hidden, params.weight_hh, params.bias_hh).chunk(3, 1);
    
    // r_t = σ(i_r + h_r)
    auto reset_gate = torch::sigmoid(chunked_igates[0] + chunked_hgates[0]);
    
    // VOLATILITY INTEGRATION: u_t = σ(i_u + h_u + γ_u * h_var)
    auto update_gate = torch::sigmoid(chunked_igates[1] + chunked_hgates[1] + gamma_u * h_var);
    
    // n_t = tanh(i_n + r_t ⊙ h_n)
    auto new_gate = torch::tanh(chunked_igates[2] + reset_gate * chunked_hgates[2]);
    
    // h_t = (h_{t-1} - n_t) ⊙ u_t + n_t
    auto new_hidden = (hidden - new_gate) * update_gate + new_gate;
    
    return new_hidden;
}

/**
 * @brief Volatility-integrated LSTM cell with volatility in forget gate
 * @param input Input tensor [batch, input_size]
 * @param hx Hidden state [batch, hidden_size]
 * @param cx Cell state [batch, hidden_size]
 * @param params LSTM parameters (weight matrices are 4*hidden_size)
 * @param garch_model GARCH model for volatility dynamics
 * @param h_var Current variance state
 * @param z_t Random shock for GARCH update
 * @param gamma_f Volatility scaling parameter
 * @param pre_compute_input Whether input is pre-computed
 * @return Tuple of (new_hidden, new_cell, new_variance)
 */
inline std::tuple<torch::Tensor, torch::Tensor> volatility_lstm_cell(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const custom::RNNParams& params,
    std::shared_ptr<GARCHModel> garch_model,
    const torch::Tensor& h_var,
    const torch::Tensor& gamma_f,
    bool pre_compute_input = false) {
    
    // Update GARCH variance
    // torch::Tensor h_var_new = garch_model->update_variance(h_var, z_t);
    
    auto ih_part = pre_compute_input ? input : 
        torch::linear(input, params.weight_ih, params.bias_ih);
    auto hh_part = torch::linear(hx, params.weight_hh, params.bias_hh);
    auto gates = ih_part + hh_part;
    
    auto chunked_gates = gates.chunk(4, 1);
    auto ingate = torch::sigmoid(chunked_gates[0]);     // i_t
    
    // VOLATILITY INTEGRATION: f_t = σ(gate + γ_f * h_var)
    auto forgetgate = torch::sigmoid(chunked_gates[1] + gamma_f * h_var); // f_t with volatility
    
    auto cellgate = torch::tanh(chunked_gates[2]);      // g_t
    auto outgate = torch::sigmoid(chunked_gates[3]);    // o_t
    
    // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    auto cy = forgetgate * cx + ingate * cellgate;
    // h_t = o_t ⊙ tanh(c_t)
    auto hy = outgate * torch::tanh(cy);
    
    return std::make_tuple(hy, cy);
}

// ============================================================================
// VOLATILITY-INTEGRATED LAYER PROCESSING
// ============================================================================

/**
 * @brief Process a single volatility-integrated GRU layer
 */
inline std::tuple<torch::Tensor, torch::Tensor> process_volatility_gru_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_hidden,  // [batch, hidden_size]
    const torch::Tensor& initial_h_var,   // [batch, 1] or scalar
    // const torch::Tensor& z_sequence,      // [seq_len, batch] random shocks
    const custom::RNNParams& params,
    std::shared_ptr<GARCHModel> garch_model,
    const torch::Tensor& gamma_u,
    const custom::RNNOptions& options = custom::RNNOptions{}) {
    
    int seq_len = input.size(0);
    int batch_size = input.size(1);
    
    std::vector<torch::Tensor> outputs;
    std::vector<torch::Tensor> variances;
    torch::Tensor hidden = initial_hidden;
    torch::Tensor h_var = initial_h_var;
    
    // Pre-compute input transformations for CPU optimization
    torch::Tensor input_transformed;
    bool pre_compute = options.pre_compute_input && input.device().is_cpu();
    if (pre_compute) {
        input_transformed = torch::linear(input, params.weight_ih, params.bias_ih);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor x_t = pre_compute ? input_transformed[t] : input[t];
        // torch::Tensor z_t = z_sequence[t];
        
        auto new_hidden = volatility_gru_cell(
            x_t, hidden, params, garch_model, h_var, gamma_u, pre_compute);
        
        hidden = new_hidden;
        
        outputs.push_back(hidden);
    }
    
    return std::make_tuple(torch::stack(outputs, 0), hidden);
}

/**
 * @brief Process a single volatility-integrated LSTM layer
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_volatility_lstm_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_h,       // [batch, hidden_size]
    const torch::Tensor& initial_c,       // [batch, hidden_size]
    const torch::Tensor& initial_h_var,   // [batch, 1] or scalar
    const custom::RNNParams& params,
    std::shared_ptr<GARCHModel> garch_model,
    const torch::Tensor& gamma_f,
    const custom::RNNOptions& options = custom::RNNOptions{}) {
    
    int seq_len = input.size(0);
    
    std::vector<torch::Tensor> outputs;
    std::vector<torch::Tensor> variances;
    torch::Tensor h = initial_h;
    torch::Tensor c = initial_c;
    torch::Tensor h_var = initial_h_var;
    
    // Pre-compute input transformations for CPU optimization
    torch::Tensor input_transformed;
    bool pre_compute = options.pre_compute_input && input.device().is_cpu();
    if (pre_compute) {
        input_transformed = torch::linear(input, params.weight_ih, params.bias_ih);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor x_t = pre_compute ? input_transformed[t] : input[t];
        
        auto [new_h, new_c] = volatility_lstm_cell(
            x_t, h, c, params, garch_model, h_var, gamma_f, pre_compute);
       
        h = new_h;
        c = new_c;
        
        outputs.push_back(h);
    }
   
    return std::make_tuple(torch::stack(outputs, 0), h, c);
}

// ============================================================================
// VOLATILITY-INTEGRATED PARAMETERS
// ============================================================================

/**
 * @brief Extended parameters for volatility-integrated RNNs
 */
struct VolatilityRNNParams {
    std::vector<custom::RNNParams> rnn_params;  // Standard RNN parameters
    std::shared_ptr<GARCHModel> garch_model;            // GARCH volatility model
    torch::Tensor gamma_f;                              // Volatility scaling parameter(s)
    torch::Tensor initial_variance;                     // Initial variance state
    torch::Tensor fc_weight;                            // Fully connected layer weight [1, hidden_size]
    torch::Tensor fc_bias;                              // Fully connected layer bias [1]
    
    VolatilityRNNParams() = default;
    
    VolatilityRNNParams(const std::vector<custom::RNNParams>& rnn_params,
                       std::shared_ptr<GARCHModel> garch_model,
                       const torch::Tensor& gamma,
                       const torch::Tensor& initial_variance = torch::tensor(0.01),
                       const torch::Tensor& fc_weight = torch::Tensor(),
                       const torch::Tensor& fc_bias = torch::Tensor())
        : rnn_params(rnn_params), garch_model(garch_model), 
          gamma_f(gamma), initial_variance(initial_variance),
          fc_weight(fc_weight), fc_bias(fc_bias) {}
};

// ============================================================================
// HIGH-LEVEL VOLATILITY-INTEGRATED FUNCTIONAL API
// ============================================================================

/**
 * @brief Volatility-integrated GRU (functional interface)
 * @param input Input tensor [seq_len, batch, input_size] or [batch, seq_len, input_size]
 * @param hx Initial hidden state [num_layers * num_directions, batch, hidden_size]
 * @param z_sequence Random shocks [seq_len, batch] for GARCH updates
 * @param vol_params Volatility-integrated parameters
 * @param options RNN options and configuration
 * @return Tuple of (output, final_hidden, variance_sequence)
 */
inline std::tuple<torch::Tensor, torch::Tensor> volatility_gru(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const VolatilityRNNParams& vol_params,
    const custom::RNNOptions& options = custom::RNNOptions{}) {
    
    // Handle batch_first format - check dimensions before transpose
    torch::Tensor x = input;
    // torch::Tensor z_seq = z_sequence;
    
    // if (options.batch_first && input.dim() >= 2) {
    //     x = input.transpose(0, 1);
    // }
    // if (options.batch_first && z_sequence.dim() >= 2) {
    //     z_seq = z_sequence.transpose(0, 1);
    // }
    
    int num_directions = options.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_outputs;
    std::vector<torch::Tensor> variance_outputs;
    
    for (int layer = 0; layer < options.num_layers; ++layer) {
        torch::Tensor layer_hidden = hx.select(0, layer * num_directions);
        
        if (options.bidirectional) {
            // TODO: Implement bidirectional volatility-integrated processing
            throw std::runtime_error("Bidirectional volatility-integrated RNN not yet implemented");
        } else {
            int idx = layer * num_directions;
            auto [layer_output, layer_final_hidden] = process_volatility_gru_layer(
                x, layer_hidden, vol_params.initial_variance,
                vol_params.rnn_params[idx], vol_params.garch_model, vol_params.gamma_f, options);
            
            layer_outputs.push_back(layer_final_hidden.unsqueeze(0));
            // variance_outputs.push_back(layer_variances);
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (options.dropout > 0.0 && options.training && layer < options.num_layers - 1) {
            x = torch::dropout(x, options.dropout, options.training);
        }
    }
    
    torch::Tensor final_hidden = torch::cat(layer_outputs, 0);
    torch::Tensor output = options.batch_first ? x.transpose(0, 1) : x;
    // torch::Tensor variance_sequence = variance_outputs.back(); // Return last layer variances
    
    return std::make_tuple(output, final_hidden);
}

/**
 * @brief Volatility-integrated LSTM (functional interface)
 * @param input Input tensor [seq_len, batch, input_size] or [batch, seq_len, input_size]
 * @param hx Initial hidden state [num_layers * num_directions, batch, hidden_size]
 * @param cx Initial cell state [num_layers * num_directions, batch, hidden_size]
 * @param z_sequence Random shocks [seq_len, batch] for GARCH updates
 * @param vol_params Volatility-integrated parameters
 * @param options RNN options and configuration
 * @return Tuple of (output, final_hidden, final_cell, variance_sequence)
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> volatility_lstm(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const VolatilityRNNParams& vol_params,
    const custom::RNNOptions& options = custom::RNNOptions{}) {
    
    // Handle batch_first format - check dimensions before transpose
    torch::Tensor x = input;
    // torch::Tensor z_seq = z_sequence;
    
    // if (options.batch_first && input.dim() >= 2) {
    //     x = input.transpose(0, 1);
    // }
    // if (options.batch_first && z_sequence.dim() >= 2) {
    //     z_seq = z_sequence.transpose(0, 1);
    // }
    
    int num_directions = options.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_h_outputs, layer_c_outputs;
    std::vector<torch::Tensor> variance_outputs;
    
    for (int layer = 0; layer < options.num_layers; ++layer) {
        if (options.bidirectional) {
            // TODO: Implement bidirectional volatility-integrated processing
            throw std::runtime_error("Bidirectional volatility-integrated RNN not yet implemented");
        } else {
            // Unidirectional LSTM processing
            int idx = layer * num_directions;
            torch::Tensor layer_h = hx.select(0, idx);
            torch::Tensor layer_c = cx.select(0, idx);
            // std::cout << "vol ht:" << vol_params.initial_variance << " at layer " << layer << "\n";
            // std::cout << "z sequence:" << z_seq << " at layer " << layer << "\n";
            auto [layer_output, layer_final_h, layer_final_c] = process_volatility_lstm_layer(
                x, layer_h, layer_c, vol_params.initial_variance, 
                vol_params.rnn_params[idx], vol_params.garch_model, vol_params.gamma_f, options);
            layer_h_outputs.push_back(layer_final_h.unsqueeze(0));
            layer_c_outputs.push_back(layer_final_c.unsqueeze(0));
            // variance_outputs.push_back(layer_variances);
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (options.dropout > 0.0 && options.training && layer < options.num_layers - 1) {
            x = torch::dropout(x, options.dropout, options.training);
        }
    }
    
    torch::Tensor final_h = torch::cat(layer_h_outputs, 0);
    torch::Tensor final_c = torch::cat(layer_c_outputs, 0);
    torch::Tensor output = options.batch_first ? x.transpose(0, 1) : x;
    // torch::Tensor variance_sequence = variance_outputs.back(); // Return last layer variances
    return std::make_tuple(output, final_h, final_c);
}

// ============================================================================
// UTILITY FUNCTIONS FOR VOLATILITY-INTEGRATED RNNS
// ============================================================================

/**
 * @brief Create volatility-integrated RNN parameters
 */
inline VolatilityRNNParams create_volatility_rnn_params(
    int input_size, int hidden_size, int num_layers, bool bidirectional,
    const std::string& garch_type = "hn",
    const std::map<std::string, double>& garch_params = {},
    int gate_multiplier = 3, // 3 for GRU, 4 for LSTM
    bool use_bias = true, 
    torch::ScalarType dtype = torch::kFloat32) {
    
    // Create standard RNN parameters
    auto rnn_params = custom::create_rnn_params_vector(
        input_size, hidden_size, num_layers, bidirectional, 
        gate_multiplier, use_bias, dtype);
    
    // Create GARCH model
    std::shared_ptr<GARCHModel> garch_model;
    torch::TensorOptions options(dtype);
    
    if (garch_type == "hn") {
        garch_model = std::make_shared<HNModel>(
            garch_params.count("omega_bar") ? garch_params.at("omega_bar") : 0.01,
            garch_params.count("alpha") ? garch_params.at("alpha") : 0.1,
            garch_params.count("phi") ? garch_params.at("phi") : 0.8,
            garch_params.count("lamda") ? garch_params.at("lamda") : 0.0,
            garch_params.count("gamma") ? garch_params.at("gamma") : 0.1,
            options);
    } else if (garch_type == "chn") {
        garch_model = std::make_shared<CHNModel>(
            garch_params.count("omega_bar") ? garch_params.at("omega_bar") : 0.01,
            garch_params.count("alpha") ? garch_params.at("alpha") : 0.1,
            garch_params.count("phi") ? garch_params.at("phi") : 0.8,
            garch_params.count("lamda") ? garch_params.at("lambda_param") : 0.0,
            garch_params.count("gamma1") ? garch_params.at("gamma1") : 0.1,
            garch_params.count("gamma2") ? garch_params.at("gamma2") : 0.05,
            garch_params.count("phi_q") ? garch_params.at("phi_q") : 0.1,
            garch_params.count("rho") ? garch_params.at("rho") : 0.9,
            options);
    } else {
        throw std::invalid_argument("garch_type must be 'hn' or 'chn'");
    }
    
    // Initialize volatility scaling parameter
    auto gamma_f = torch::tensor(1.0, options);
    auto initial_variance = torch::tensor(0.01, options);
    
    // Initialize fully connected layer parameters
    // FC layer maps from hidden_size to 1 (scalar output)
    double std_val = 1.0 / std::sqrt(hidden_size);
    auto fc_weight = torch::empty({1, hidden_size}, options);
    auto fc_bias = torch::empty({1}, options);
    torch::nn::init::uniform_(fc_weight, -std_val, std_val);
    torch::nn::init::uniform_(fc_bias, -std_val, std_val);
    
    return VolatilityRNNParams(rnn_params, garch_model, gamma_f, initial_variance, fc_weight, fc_bias);
}

/**
 * @brief Generate random shocks for GARCH updates
 */
inline torch::Tensor generate_garch_shocks(int seq_len, int batch_size, 
                                          const torch::TensorOptions& options = torch::kFloat32) {
    return torch::randn({seq_len, batch_size}, options);
}

/**
 * @brief Factory function to create GARCH models from string identifier
 * @param model_type Model type identifier ("hn", "chn", etc.)
 * @param params Parameter configuration map
 * @param options Tensor options for parameter initialization
 * @return Shared pointer to the created GARCH model
 */
inline std::shared_ptr<GARCHModel> create_garch_model(
    const std::string& model_type,
    const std::map<std::string, double>& params,
    const torch::TensorOptions& options = torch::kFloat32) {
    
    if (model_type == "hn") {
        return std::make_shared<HNModel>(
            params.count("omega_bar") ? params.at("omega_bar") : 0.01,
            params.count("alpha") ? params.at("alpha") : 0.1,
            params.count("phi") ? params.at("phi") : 0.8,
            params.count("lamda") ? params.at("lamda") : 0.0,
            params.count("gamma") ? params.at("gamma") : 0.1,
            options);
    } else if (model_type == "chn") {
        return std::make_shared<CHNModel>(
            params.count("omega_bar") ? params.at("omega_bar") : 0.01,
            params.count("alpha") ? params.at("alpha") : 0.1,
            params.count("phi") ? params.at("phi") : 0.8,
            params.count("lamda") ? params.at("lamda") : 0.0,
            params.count("gamma1") ? params.at("gamma1") : 0.1,
            params.count("gamma2") ? params.at("gamma2") : 0.05,
            params.count("phi_q") ? params.at("phi_q") : 0.1,
            params.count("rho") ? params.at("rho") : 0.9,
            options);
    } else {
        throw std::invalid_argument("Unknown GARCH model type: " + model_type + 
                                  ". Supported types: 'hn', 'chn'");
    }
}

/**
 * @brief Collect all learnable parameters from volatility-integrated RNN
 * @param vol_params Volatility-integrated RNN parameters
 * @return Vector of all learnable tensors
 */
inline std::vector<torch::Tensor> collect_all_parameters(const VolatilityRNNParams& vol_params) {
    std::vector<torch::Tensor> parameters;
    
    // Add RNN parameters
    parameters.push_back(vol_params.gamma_f);
    parameters.push_back(vol_params.fc_weight);
    parameters.push_back(vol_params.fc_bias);
    
    for (const auto& param_set : vol_params.rnn_params) {
        parameters.push_back(param_set.weight_ih);
        parameters.push_back(param_set.weight_hh);
        if (param_set.has_bias()) {
            parameters.push_back(param_set.bias_ih);
            parameters.push_back(param_set.bias_hh);
        }
    }
    
    // Add GARCH parameters using unified interface
    auto garch_params = vol_params.garch_model->get_learnable_parameters();
    parameters.insert(parameters.end(), garch_params.begin(), garch_params.end());
    
    return parameters;
}

/**
 * @brief Set all parameters as learnable in volatility-integrated RNN
 * @param vol_params Volatility-integrated RNN parameters
 * @param learnable Whether parameters should require gradients
 */
inline void set_all_parameters_learnable(VolatilityRNNParams& vol_params, bool learnable = true) {
    // Set RNN parameters learnable
    vol_params.gamma_f.set_requires_grad(learnable);
    vol_params.fc_weight.set_requires_grad(learnable);
    vol_params.fc_bias.set_requires_grad(learnable);
    
    for (auto& param_set : vol_params.rnn_params) {
        param_set.weight_ih.set_requires_grad(learnable);
        param_set.weight_hh.set_requires_grad(learnable);
        if (param_set.has_bias()) {
            param_set.bias_ih.set_requires_grad(learnable);
            param_set.bias_hh.set_requires_grad(learnable);
        }
    }
    
    // Set GARCH parameters learnable using unified interface
    vol_params.garch_model->set_parameters_learnable(learnable);
}

// ============================================================================
// MODEL SERIALIZATION FUNCTIONS
// ============================================================================

/**
 * @brief Save complete volatility-integrated model to file
 * @param vol_params Complete model parameters (RNN + GARCH + FC)
 * @param model_type RNN type ("gru" or "lstm")
 * @param filepath Path to save the model
 * @param metadata Additional metadata to save with model
 */
inline void save_volatility_model(
    const VolatilityRNNParams& vol_params,
    const std::string& model_type,
    const std::string& filepath,
    const std::map<std::string, std::string>& metadata = {}) {
    
    // Create a vector of tensors to save (PyTorch supports this)
    std::vector<torch::Tensor> tensors_to_save;
    std::vector<std::string> tensor_names;
    
    // Save RNN parameters
    tensors_to_save.push_back(vol_params.gamma_f);
    tensor_names.push_back("gamma_f");
    
    tensors_to_save.push_back(vol_params.initial_variance);
    tensor_names.push_back("initial_variance");
    
    tensors_to_save.push_back(vol_params.fc_weight);
    tensor_names.push_back("fc_weight");
    
    tensors_to_save.push_back(vol_params.fc_bias);
    tensor_names.push_back("fc_bias");
    
    // Save RNN layer parameters
    for (size_t i = 0; i < vol_params.rnn_params.size(); ++i) {
        const auto& param_set = vol_params.rnn_params[i];
        std::string layer_prefix = "rnn_layer_" + std::to_string(i) + "_";
        
        tensors_to_save.push_back(param_set.weight_ih);
        tensor_names.push_back(layer_prefix + "weight_ih");
        
        tensors_to_save.push_back(param_set.weight_hh);
        tensor_names.push_back(layer_prefix + "weight_hh");
        
        if (param_set.has_bias()) {
            tensors_to_save.push_back(param_set.bias_ih);
            tensor_names.push_back(layer_prefix + "bias_ih");
            
            tensors_to_save.push_back(param_set.bias_hh);
            tensor_names.push_back(layer_prefix + "bias_hh");
        }
    }
    
    // Save GARCH parameters using unified interface
    auto garch_state = vol_params.garch_model->get_state_dict();
    for (const auto& [key, tensor] : garch_state) {
        tensors_to_save.push_back(tensor);
        tensor_names.push_back("garch_" + key);
    }
    
    // Save model metadata as tensors
    tensors_to_save.push_back(torch::tensor(model_type == "lstm" ? 1.0 : 0.0));
    tensor_names.push_back("_model_type");
    
    tensors_to_save.push_back(torch::tensor(vol_params.garch_model->get_model_name() == "CHN" ? 1.0 : 0.0));
    tensor_names.push_back("_garch_type");
    
    tensors_to_save.push_back(torch::tensor(static_cast<double>(vol_params.rnn_params.size())));
    tensor_names.push_back("_num_layers");
    
    if (!vol_params.rnn_params.empty()) {
        tensors_to_save.push_back(torch::tensor(static_cast<double>(vol_params.rnn_params[0].weight_ih.size(1))));
        tensor_names.push_back("_input_size");
        
        tensors_to_save.push_back(torch::tensor(static_cast<double>(vol_params.rnn_params[0].weight_hh.size(1))));
        tensor_names.push_back("_hidden_size");
        
        tensors_to_save.push_back(torch::tensor(vol_params.rnn_params[0].has_bias() ? 1.0 : 0.0));
        tensor_names.push_back("_has_bias");
    }
    
    // Save tensors and names separately
    torch::save(tensors_to_save, filepath);
    
    // Save tensor names to a separate text file
    std::string names_file = filepath.substr(0, filepath.find_last_of('.')) + "_names.txt";
    std::ofstream names_stream(names_file);
    for (const auto& name : tensor_names) {
        names_stream << name << "\n";
    }
    names_stream.close();
    
    std::cout << "Model saved to: " << filepath << std::endl;
    std::cout << "Model type: " << model_type << "-" << vol_params.garch_model->get_model_name() << std::endl;
    std::cout << "Total parameters: " << collect_all_parameters(vol_params).size() << std::endl;
}

/**
 * @brief Load complete volatility-integrated model from file
 * @param filepath Path to the saved model file
 * @return Tuple of (VolatilityRNNParams, model_type, garch_type)
 */
inline std::tuple<VolatilityRNNParams, std::string, std::string> load_volatility_model(
    const std::string& filepath) {
    
    // Load tensors
    std::vector<torch::Tensor> loaded_tensors;
    torch::load(loaded_tensors, filepath);
    
    // Load tensor names
    std::string names_file = filepath.substr(0, filepath.find_last_of('.')) + "_names.txt";
    std::vector<std::string> tensor_names;
    std::ifstream names_stream(names_file);
    std::string name;
    while (std::getline(names_stream, name)) {
        tensor_names.push_back(name);
    }
    names_stream.close();
    
    // Create a map for easy access
    std::map<std::string, torch::Tensor> complete_state_dict;
    for (size_t i = 0; i < tensor_names.size() && i < loaded_tensors.size(); ++i) {
        complete_state_dict[tensor_names[i]] = loaded_tensors[i];
    }
    
    // Extract metadata
    bool is_lstm = complete_state_dict["_model_type"].item<double>() > 0.5;
    bool is_chn = complete_state_dict["_garch_type"].item<double>() > 0.5;
    int num_layers = static_cast<int>(complete_state_dict["_num_layers"].item<double>());
    int input_size = static_cast<int>(complete_state_dict["_input_size"].item<double>());
    int hidden_size = static_cast<int>(complete_state_dict["_hidden_size"].item<double>());
    bool has_bias = complete_state_dict["_has_bias"].item<double>() > 0.5;
    
    std::string model_type = is_lstm ? "lstm" : "gru";
    std::string garch_type = is_chn ? "chn" : "hn";
    
    // Create RNN parameters structure
    int gate_multiplier = is_lstm ? 4 : 3;
    auto rnn_params = custom::create_rnn_params_vector(
        input_size, hidden_size, num_layers, false, 
        gate_multiplier, has_bias, torch::kFloat32);
    
    // Load RNN layer parameters
    for (int i = 0; i < num_layers; ++i) {
        std::string layer_prefix = "rnn_layer_" + std::to_string(i) + "_";
        
        rnn_params[i].weight_ih = complete_state_dict[layer_prefix + "weight_ih"];
        rnn_params[i].weight_hh = complete_state_dict[layer_prefix + "weight_hh"];
        
        if (has_bias) {
            rnn_params[i].bias_ih = complete_state_dict[layer_prefix + "bias_ih"];
            rnn_params[i].bias_hh = complete_state_dict[layer_prefix + "bias_hh"];
        }
    }
    
    // Create GARCH model and load its parameters
    std::shared_ptr<GARCHModel> garch_model;
    if (is_chn) {
        garch_model = std::make_shared<CHNModel>(0.01, 0.1, 0.8, 0.0, 0.1, 0.05, 0.1, 0.9);
    } else {
        garch_model = std::make_shared<HNModel>(0.01, 0.1, 0.8, 0.0, 0.1);
    }
    
    // Extract GARCH parameters
    std::map<std::string, torch::Tensor> garch_state;
    for (const auto& [key, tensor] : complete_state_dict) {
        if (key.substr(0, 6) == "garch_") {
            garch_state[key.substr(6)] = tensor;
        }
    }
    garch_model->load_state_dict(garch_state);
    
    // Load other parameters
    auto gamma_f = complete_state_dict["gamma_f"];
    auto initial_variance = complete_state_dict["initial_variance"];
    auto fc_weight = complete_state_dict["fc_weight"];
    auto fc_bias = complete_state_dict["fc_bias"];
    
    // Create VolatilityRNNParams
    VolatilityRNNParams vol_params(rnn_params, garch_model, gamma_f, initial_variance, fc_weight, fc_bias);
    
    std::cout << "Model loaded from: " << filepath << std::endl;
    std::cout << "Model type: " << model_type << "-" << garch_type << std::endl;
    std::cout << "Architecture: " << input_size << " -> " << hidden_size << " (" << num_layers << " layers)" << std::endl;
    
    return std::make_tuple(vol_params, model_type, garch_type);
}

/**
 * @brief Save model training results and metadata
 * @param results Training results (losses, log-likelihoods, etc.)
 * @param model_name Model identifier
 * @param filepath Path to save results
 */
inline void save_training_results(
    const std::map<std::string, std::vector<double>>& results,
    const std::string& model_name,
    const std::string& filepath) {
    
    // Create vectors for tensors and names (same approach as model saving)
    std::vector<torch::Tensor> tensors_to_save;
    std::vector<std::string> tensor_names;
    
    for (const auto& [key, values] : results) {
        auto tensor = torch::zeros({static_cast<long>(values.size())});
        for (size_t i = 0; i < values.size(); ++i) {
            tensor[i] = values[i];
        }
        tensors_to_save.push_back(tensor);
        tensor_names.push_back(key);
    }
    
    // Add metadata
    auto model_name_tensor = torch::zeros({static_cast<long>(model_name.length())});
    for (size_t i = 0; i < model_name.length(); ++i) {
        model_name_tensor[i] = static_cast<double>(model_name[i]);
    }
    tensors_to_save.push_back(model_name_tensor);
    tensor_names.push_back("_model_name");
    
    // Save tensors and names separately
    torch::save(tensors_to_save, filepath);
    
    // Save tensor names to a separate text file
    std::string names_file = filepath.substr(0, filepath.find_last_of('.')) + "_names.txt";
    std::ofstream names_stream(names_file);
    for (const auto& name : tensor_names) {
        names_stream << name << "\n";
    }
    names_stream.close();
    
    std::cout << "Training results saved to: " << filepath << std::endl;
}

/**
 * @brief Create a trained model package with all necessary components
 * @param vol_params Trained model parameters
 * @param model_type RNN type
 * @param training_results Training history
 * @param base_filename Base name for saved files
 */
inline void save_model_package(
    const VolatilityRNNParams& vol_params,
    const std::string& model_type,
    const std::map<std::string, std::vector<double>>& training_results,
    const std::string& base_filename) {
    
    std::string model_name = model_type + "_" + vol_params.garch_model->get_model_name();
    
    // Save model parameters
    save_volatility_model(vol_params, model_type, base_filename + "_model.pt");
    
    // Save training results
    save_training_results(training_results, model_name, base_filename + "_training.pt");
    
    // Save model summary as text
    std::ofstream summary_file(base_filename + "_summary.txt");
    summary_file << "Volatility-Integrated Model Summary\n";
    summary_file << "===================================\n";
    summary_file << "Model Type: " << model_type << "-" << vol_params.garch_model->get_model_name() << "\n";
    summary_file << "Total Parameters: " << collect_all_parameters(vol_params).size() << "\n";
    
    if (!vol_params.rnn_params.empty()) {
        summary_file << "Input Size: " << vol_params.rnn_params[0].weight_ih.size(1) << "\n";
        summary_file << "Hidden Size: " << vol_params.rnn_params[0].weight_hh.size(1) << "\n";
        summary_file << "Number of Layers: " << vol_params.rnn_params.size() << "\n";
        summary_file << "Has Bias: " << (vol_params.rnn_params[0].has_bias() ? "Yes" : "No") << "\n";
    }
    
    if (training_results.count("log_likelihoods") && !training_results.at("log_likelihoods").empty()) {
        summary_file << "Final Log-Likelihood: " << training_results.at("log_likelihoods").back() << "\n";
    }
    
    summary_file.close();
    
    std::cout << "Complete model package saved with base name: " << base_filename << std::endl;
}

// ============================================================================
// SIMPLIFIED INTERFACE
// ============================================================================

namespace simple {

/**
 * @brief Simplified volatility-integrated GRU
 */
inline std::tuple<torch::Tensor, torch::Tensor> volatility_gru(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const VolatilityRNNParams& vol_params,
    bool has_bias = true,
    int num_layers = 1,
    double dropout = 0.0,
    bool training = true,
    bool bidirectional = false,
    bool batch_first = true) {
    
    custom::RNNOptions options;
    options.use_bias = has_bias;
    options.num_layers = num_layers;
    options.dropout = dropout;
    options.training = training;
    options.bidirectional = bidirectional;
    options.batch_first = batch_first;
    
    return volatility_rnn_functional::volatility_gru(input, hx, vol_params, options);
}

/**
 * @brief Simplified volatility-integrated LSTM
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> volatility_lstm(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    // const torch::Tensor& h_t,
    const VolatilityRNNParams& vol_params,
    bool has_bias = true,
    int num_layers = 1,
    double dropout = 0.0,
    bool training = true,
    bool bidirectional = false,
    bool batch_first = false) {
    
    custom::RNNOptions options;
    options.use_bias = has_bias;
    options.num_layers = num_layers;
    options.dropout = dropout;
    options.training = training;
    options.bidirectional = bidirectional;
    options.batch_first = batch_first;
    
    return volatility_rnn_functional::volatility_lstm(input, hx, cx, vol_params, options);
    
}

} // namespace simple

} // namespace volatility_rnn_functional
