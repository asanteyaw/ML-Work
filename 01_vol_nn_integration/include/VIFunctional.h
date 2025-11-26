#pragma once

#include <cstdint>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include <tuple>
#include <span>
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
namespace vol_rnn_functional {

// ============================================================================
// NETWORK WEIGHT STRUCTURE
// ============================================================================

struct NetWeight {
    torch::Tensor weight_ih{}, weight_hh{}, bias_ih{}, bias_hh{};
    torch::Tensor fc_weight{}, fc_bias{};

    NetWeight() = default;
    
    NetWeight(const torch::Tensor& w_ih, const torch::Tensor& w_hh,
              const torch::Tensor& b_ih = torch::Tensor(),
              const torch::Tensor& b_hh = torch::Tensor(),
              const torch::Tensor& fc_w = torch::Tensor(),
              const torch::Tensor& fc_b = torch::Tensor())
        : weight_ih(w_ih), weight_hh(w_hh), bias_ih(b_ih), bias_hh(b_hh), fc_weight(fc_w), fc_bias(fc_b) {}
    
    bool has_bias() const {
        return bias_ih.defined() && bias_hh.defined();
    }

    bool has_fc() const {
        return fc_weight.defined();
    }

    torch::Tensor rec_weights (){
        return torch::stack({weight_ih.flatten(), weight_hh.flatten(), bias_ih, bias_hh});
    }

    torch::Tensor dense_weights (){
       
        TORCH_CHECK(has_fc(),  "Did not provide weights for fully connected layer");
        return torch::stack({fc_weight.flatten(), fc_bias});
    }

    int64_t rec_weight_count (){
        return rec_weights().numel();
    }

    int64_t fc_weight_count (){
        return dense_weights().numel();
    }

    int64_t agg_net_weight (){
        return torch::stack({rec_weights(), dense_weights()}).numel();
    }

};

// ============================================================================
// NETWORK WEIGHT STRUCTURE
// ============================================================================
struct NetConfig {
    bool training = true;                    // Training mode (affects dropout)
    bool batch_first = true;               // Input format: true -> (batch, seq, features) vs false -> (seq, batch, features)
    bool bidirectional = false;             // Bidirectional processing
    double dropout = 0.0;                   // Dropout probability between layers
    int n_gates = 3;                        // Number of gates 3 for GRU, 4 for LSTM
    int in_size = 1;                        // Input size
    int hid_size = 1;                       // Hidden size
    int out_size = 1;                       // Output size, typically requires FC layer
    int num_layers = 1;                     // Number of layers
    bool proj_size = 0;                     // Projection size for LSTM (0 = no projection)
    bool use_bias = true;                   // Whether to use bias parameters
    
    // Advanced options
    bool pre_compute_input = true;          // Pre-compute input transformations (CPU optimization)
    bool force_cpu_fallback = false;       // Force CPU implementation even on CUDA
    std::optional<torch::ScalarType> dtype = std::nullopt;  // Force specific dtype
    std::optional<torch::Device> device = std::nullopt;     // Force specific device
    
    // Numerical stability options
    double gradient_clip_value = 0.0;      // Gradient clipping (0 = disabled)
    bool use_layer_norm = false;           // Apply layer normalization
    double epsilon = 1e-5;                 // Epsilon for numerical stability
    
    NetConfig() = default;
};

// ============================================================================
// NEURAL STEP ARCHIVE
// ============================================================================
struct Shelf {
    // std::vector<NetWeight> rnn_params;
    std::span<NetWeight> rnn_params;
    torch::Tensor gamma_f;
    torch::Tensor h_t;

    Shelf() = default;
};

// ============================================================================
// VOLATILITY-INTEGRATED CELL FUNCTIONS
// ============================================================================

/**
 * @brief Volatility-integrated GRU cell with volatility in update gate
 * @param input Input tensor [batch, input_size]
 * @param hidden Hidden state [batch, hidden_size]
 * @param params GRU parameters (weight matrices are 3*hidden_size)
 * @param h_var Current variance state
 * @param gamma_u Volatility scaling parameter
 * @param pre_compute_input Whether input is pre-computed
 * @return Tuple of (new_hidden, new_variance)
 */
inline torch::Tensor vol_gru_cell(
    const torch::Tensor& input,
    const torch::Tensor& hidden,
    const NetWeight& params,
    const torch::Tensor& h_var,
    const torch::Tensor& gamma_u,
    bool pre_compute_input = false) {

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
    auto eta_t = (hidden - new_gate) * update_gate + new_gate;
    
    return eta_t;
}

/**
 * @brief Volatility-integrated LSTM cell with volatility in forget gate
 * @param input Input tensor [batch, input_size]
 * @param hx Hidden state [batch, hidden_size]
 * @param cx Cell state [batch, hidden_size]
 * @param params LSTM parameters (weight matrices are 4*hidden_size)
 * @param h_var Current variance state
 * @param gamma_f Volatility scaling parameter
 * @param pre_compute_input Whether input is pre-computed
 * @return Tuple of (new_hidden, new_cell, new_variance)
 */
inline std::tuple<torch::Tensor, torch::Tensor> vol_lstm_cell(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const NetWeight& params,
    const torch::Tensor& h_var,
    const torch::Tensor& gamma_f,
    bool pre_compute_input = false) {
    
    auto ih_part = pre_compute_input ? input : 
        torch::linear(input, params.weight_ih, params.bias_ih);
    auto hh_part = torch::linear(hx, params.weight_hh, params.bias_hh);
    auto gates = ih_part + hh_part;
    
    auto chunked_gates = gates.chunk(4, 1);
    auto in_gate = torch::sigmoid(chunked_gates[0]);     // i_t
    
    // VOLATILITY INTEGRATION: f_t = σ(gate + γ_f * h_var)
    auto forget_gate = torch::sigmoid(chunked_gates[1] + gamma_f * h_var); // f_t with volatility
    
    auto cell_gate = torch::tanh(chunked_gates[2]);      // g_t
    auto out_gate = torch::sigmoid(chunked_gates[3]);    // o_t
    
    // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    auto c_t = forget_gate * cx + in_gate * cell_gate;
    // η_t = o_t ⊙ tanh(c_t)
    auto eta_t = out_gate * torch::tanh(c_t);
    
    return std::make_tuple(eta_t, c_t);
}

// ============================================================================
// VOLATILITY-INTEGRATED LAYER PROCESSING
// ============================================================================

/**
 * @brief Process a single volatility-integrated GRU layer
 */
inline std::tuple<torch::Tensor, torch::Tensor> vol_gru_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_hidden,  // [batch, hidden_size]
    const torch::Tensor& init_h_var,   // [batch, 1] or scalar
    // const torch::Tensor& z_sequence,      // [seq_len, batch] random shocks
    const NetWeight& params,
    const torch::Tensor& gamma_u,
    const NetConfig& config = NetConfig{}) {
    
    int seq_len = input.size(0);
    int batch_size = input.size(1);
    
    std::vector<torch::Tensor> outputs;
    std::vector<torch::Tensor> variances;
    torch::Tensor hidden = initial_hidden;
    torch::Tensor h_var = init_h_var;
    
    // Pre-compute input transformations for CPU optimization
    torch::Tensor input_transformed;
    bool pre_compute = config.pre_compute_input && input.device().is_cpu();
    if (pre_compute) {
        input_transformed = torch::linear(input, params.weight_ih, params.bias_ih);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor x_t = pre_compute ? input_transformed[t] : input[t];
        // torch::Tensor z_t = z_sequence[t];
        
        auto h_ = vol_gru_cell(x_t, hidden, params, h_var, gamma_u, pre_compute);
        
        hidden = h_;
        
        outputs.push_back(hidden);
    }
    
    return std::make_tuple(torch::stack(outputs, 0), hidden);
}

/**
 * @brief Process a single volatility-integrated LSTM layer
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> vol_lstm_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_h,       // [batch, hidden_size]
    const torch::Tensor& initial_c,       // [batch, hidden_size]
    const torch::Tensor& init_h_var,   // [batch, 1] or scalar
    const NetWeight& params,
    const torch::Tensor& gamma_f,
    const NetConfig& config = NetConfig{}) {
    
    int seq_len = input.size(0);
    
    std::vector<torch::Tensor> outputs;
    std::vector<torch::Tensor> variances;
    torch::Tensor h = initial_h;
    torch::Tensor c = initial_c;
    torch::Tensor h_var = init_h_var;
    
    // Pre-compute input transformations for CPU optimization
    torch::Tensor input_transformed;
    bool pre_compute = config.pre_compute_input && input.device().is_cpu();
    if (pre_compute) {
        input_transformed = torch::linear(input, params.weight_ih, params.bias_ih);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor x_t = pre_compute ? input_transformed[t] : input[t];
        
        auto [new_h, new_c] = vol_lstm_cell(
            x_t, h, c, params, h_var, gamma_f, pre_compute);
       
        h = new_h;
        c = new_c;
        
        outputs.push_back(h);
    }
   
    return std::make_tuple(torch::stack(outputs, 0), h, c);
}

// ============================================================================
// HIGH-LEVEL VOLATILITY-INTEGRATED FUNCTIONAL API
// ============================================================================

/**
 * @brief Volatility-integrated GRU (functional interface)
 * @param input Input tensor [seq_len, batch, input_size] or [batch, seq_len, input_size]
 * @param hx Initial hidden state [num_layers * num_directions, batch, hidden_size]
 * @param vol_params Volatility-integrated parameters
 * @param config Network configuration
 * @return Tuple of (output, final_hidden)
 */
inline std::tuple<torch::Tensor, torch::Tensor> volatility_gru(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const Shelf& vi_info,
    const NetConfig& config = NetConfig{}) {
    
    // Handle batch_first format - check dimensions before transpose
    torch::Tensor x = input;
    // torch::Tensor z_seq = z_sequence;
    
    // if (options.batch_first && input.dim() >= 2) {
    //     x = input.transpose(0, 1);
    // }
    // if (options.batch_first && z_sequence.dim() >= 2) {
    //     z_seq = z_sequence.transpose(0, 1);
    // }
    
    int num_directions = config.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_outputs;
    std::vector<torch::Tensor> variance_outputs;
    
    for (int layer = 0; layer < config.num_layers; ++layer) {
        torch::Tensor layer_hidden = hx.select(0, layer * num_directions);
        
        if (config.bidirectional) {
            // TODO: Implement bidirectional volatility-integrated processing
            throw std::runtime_error("Bidirectional volatility-integrated RNN not yet implemented");
        } else {
            int idx = layer * num_directions;
            auto [layer_output, layer_final_hidden] = vol_gru_layer(
                x, layer_hidden, vi_info.h_t,
                vi_info.rnn_params[idx], vi_info.gamma_f, config);
            
            layer_outputs.push_back(layer_final_hidden.unsqueeze(0));
    
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (config.dropout > 0.0 && config.training && layer < config.num_layers - 1) {
            x = torch::dropout(x, config.dropout, config.training);
        }
    }
    
    torch::Tensor final_hidden = torch::cat(layer_outputs, 0);
    torch::Tensor output = config.batch_first ? x.transpose(0, 1) : x;
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
 * @param config Network configuration
 * @return Tuple of (output, final_hidden, final_cell, variance_sequence)
 */
inline std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> volatility_lstm(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const Shelf& vi_info,
    const NetConfig& config = NetConfig{}) {
    
    // Handle batch_first format - check dimensions before transpose
    torch::Tensor x = input;
    // torch::Tensor z_seq = z_sequence;
    
    // if (options.batch_first && input.dim() >= 2) {
    //     x = input.transpose(0, 1);
    // }
    // if (options.batch_first && z_sequence.dim() >= 2) {
    //     z_seq = z_sequence.transpose(0, 1);
    // }
    
    int num_directions = config.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_h_outputs, layer_c_outputs;
    std::vector<torch::Tensor> variance_outputs;
    
    for (int layer = 0; layer < config.num_layers; ++layer) {
        if (config.bidirectional) {
            // TODO: Implement bidirectional volatility-integrated processing
            throw std::runtime_error("Bidirectional volatility-integrated RNN not yet implemented");
        } else {
            // Unidirectional LSTM processing
            int idx = layer * num_directions;
            torch::Tensor layer_h = hx.select(0, idx);
            torch::Tensor layer_c = cx.select(0, idx);
            
            auto [layer_output, layer_final_h, layer_final_c] = vol_lstm_layer(
                x, layer_h, layer_c, vi_info.h_t, 
                vi_info.rnn_params[idx], vi_info.gamma_f, config);

            layer_h_outputs.push_back(layer_final_h.unsqueeze(0));
            layer_c_outputs.push_back(layer_final_c.unsqueeze(0));
            
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (config.dropout > 0.0 && config.training && layer < config.num_layers - 1) {
            x = torch::dropout(x, config.dropout, config.training);
        }
    }
    
    torch::Tensor final_h = torch::cat(layer_h_outputs, 0);
    torch::Tensor final_c = torch::cat(layer_c_outputs, 0);
    torch::Tensor output = config.batch_first ? x.transpose(0, 1) : x;
    // torch::Tensor variance_sequence = variance_outputs.back(); // Return last layer variances
    return std::make_pair(output, std::make_tuple(final_c, final_h));
}

inline NetWeight make_params(int input_size, int hidden_size, int gate_multiplier = 1, 
                                  bool use_bias = true, torch::ScalarType dtype = torch::kFloat32) {
    double std_val = 1.0 / std::sqrt(hidden_size);
    
    auto w_ih = torch::empty({gate_multiplier * hidden_size, input_size}, dtype);
    auto w_hh = torch::empty({gate_multiplier * hidden_size, hidden_size}, dtype);
    torch::nn::init::uniform_(w_ih, -std_val, std_val);
    torch::nn::init::uniform_(w_hh, -std_val, std_val);
    
    torch::Tensor b_ih, b_hh;
    if (use_bias) {
        b_ih = torch::empty({gate_multiplier * hidden_size}, dtype);
        b_hh = torch::empty({gate_multiplier * hidden_size}, dtype);
        torch::nn::init::uniform_(b_ih, -std_val, std_val);
        torch::nn::init::uniform_(b_hh, -std_val, std_val);
    }
    
    return NetWeight(w_ih, w_hh, b_ih, b_hh);
}

/**
 * @brief Create parameter vector for multi-layer RNN
 */
inline std::vector<NetWeight> make_params(NetConfig& config, int gate_multiplier = 1) {
    
    std::vector<NetWeight> params;
    int num_directions = config.bidirectional ? 2 : 1;
    
    for (int layer = 0; layer < config.num_layers; ++layer) {
        int layer_input_size = (layer == 0) ? config.in_size : config.hid_size * num_directions;
        
        for (int direction = 0; direction < num_directions; ++direction) {
            params.push_back(make_params(layer_input_size, config.hid_size, 
                                             gate_multiplier, config.use_bias));
        }
    }
    
    return params;
}

/**
 * @brief Simplified volatility-integrated GRU/LSTM interface
 */
inline std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> neural_step(
    const torch::Tensor& input,
    const Shelf& vi_info,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    NetConfig config) 
{
    
    if (config.n_gates == 3){
        auto [out, hid] = volatility_gru(input, hx, vi_info, config);
        // std::cout << "About to return in neural step\n";
        return std::make_pair(out, std::make_tuple(torch::tensor(0.0), hid));
    }else {
        return volatility_lstm(input, hx, cx, vi_info, config);
        
    }
}

/**
 * @brief Collect a vector of tensor and make them into NetWeights
 * @param params Parameter vector of tensors 
 * @return A vector of NetWeights
 */

 inline std::vector<NetWeight> make_net_weights(std::vector<torch::Tensor> params, NetConfig& config){
    std::vector<NetWeight> weights;
    int64_t idx{0};
    if (config.use_bias) {
        for (int layer = 0; layer < config.num_layers; ++layer) {
            auto w_ih = params.at(idx);
            auto w_hh = params.at(++idx);
            auto b_ih = params.at(++idx);
            auto b_hh = params.at(++idx);

            weights.push_back(NetWeight(w_ih, w_hh, b_ih, b_hh));
        }
    }else{
        for (int layer = 0; layer < config.num_layers; ++layer) {
            auto w_ih = params.at(idx);
            auto w_hh = params.at(++idx);

            weights.push_back(NetWeight(w_ih, w_hh));
        }
    }

    return weights;
 }

} // namespace vol_rnn_functional

