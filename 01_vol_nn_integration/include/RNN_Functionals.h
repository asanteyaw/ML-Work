#pragma once

#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <optional>

/**
 * @brief Low-level functional RNN operations matching PyTorch's functional API
 * 
 * This namespace provides functional-style RNN operations that exactly match
 * PyTorch's CPU implementations with maximum flexibility and control.
 * 
 * Usage:
 * ```cpp
 * auto [output, hidden] = custom::gru(
 *     input, hx, weights, biases, 
 *     num_layers, dropout, training, bidirectional, batch_first
 * );
 * ```
 */
namespace custom {

// ============================================================================
// PARAMETER STRUCTURES
// ============================================================================

/**
 * @brief RNN parameter pack for a single layer and direction
 */
struct RNNParams {
    torch::Tensor weight_ih;  // Input-to-hidden weight [gate_size, input_size]
    torch::Tensor weight_hh;  // Hidden-to-hidden weight [gate_size, hidden_size]
    torch::Tensor bias_ih;    // Input-to-hidden bias [gate_size] (optional)
    torch::Tensor bias_hh;    // Hidden-to-hidden bias [gate_size] (optional)
    
    RNNParams() = default;
    
    RNNParams(const torch::Tensor& w_ih, const torch::Tensor& w_hh,
              const torch::Tensor& b_ih = torch::Tensor(),
              const torch::Tensor& b_hh = torch::Tensor())
        : weight_ih(w_ih), weight_hh(w_hh), bias_ih(b_ih), bias_hh(b_hh) {}
    
    bool has_bias() const {
        return bias_ih.defined() && bias_hh.defined();
    }
};

/**
 * @brief Extended RNN options for maximum flexibility
 */
struct RNNOptions {
    bool training = true;                    // Training mode (affects dropout)
    bool batch_first = false;               // Input format: (batch, seq, features) vs (seq, batch, features)
    bool bidirectional = false;             // Bidirectional processing
    double dropout = 0.0;                   // Dropout probability between layers
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
    
    RNNOptions() = default;
};

// ============================================================================
// CORE CELL FUNCTIONS
// ============================================================================

/**
 * @brief Single RNN cell with Tanh activation
 * @param input Input tensor [batch, input_size]
 * @param hidden Hidden state [batch, hidden_size]
 * @param params RNN parameters
 * @param pre_compute_input Whether input is pre-computed
 * @return New hidden state [batch, hidden_size]
 */
inline torch::Tensor rnn_tanh_cell(
    const torch::Tensor& input,
    const torch::Tensor& hidden,
    const RNNParams& params,
    bool pre_compute_input = false) {
    
    auto ih_part = pre_compute_input ? input : 
        torch::linear(input, params.weight_ih, params.bias_ih);
    auto hh_part = torch::linear(hidden, params.weight_hh, params.bias_hh);
    return torch::tanh(ih_part + hh_part);
}

/**
 * @brief Single RNN cell with ReLU activation
 */
inline torch::Tensor rnn_relu_cell(
    const torch::Tensor& input,
    const torch::Tensor& hidden,
    const RNNParams& params,
    bool pre_compute_input = false) {
    
    auto ih_part = pre_compute_input ? input : 
        torch::linear(input, params.weight_ih, params.bias_ih);
    auto hh_part = torch::linear(hidden, params.weight_hh, params.bias_hh);
    return torch::relu(ih_part + hh_part);
}

/**
 * @brief Single GRU cell
 * @param input Input tensor [batch, input_size]
 * @param hidden Hidden state [batch, hidden_size]
 * @param params GRU parameters (weight matrices are 3*hidden_size)
 * @param pre_compute_input Whether input is pre-computed
 * @return New hidden state [batch, hidden_size]
 */
inline torch::Tensor gru_cell(
    const torch::Tensor& input,
    const torch::Tensor& hidden,
    const RNNParams& params,
    bool pre_compute_input = false) {
    
    auto chunked_igates = pre_compute_input ? 
        input.chunk(3, 1) : 
        torch::linear(input, params.weight_ih, params.bias_ih).chunk(3, 1);
    auto chunked_hgates = torch::linear(hidden, params.weight_hh, params.bias_hh).chunk(3, 1);
    
    // r_t = σ(i_r + h_r)
    auto reset_gate = torch::sigmoid(chunked_igates[0] + chunked_hgates[0]);
    // z_t = σ(i_i + h_i) 
    auto input_gate = torch::sigmoid(chunked_igates[1] + chunked_hgates[1]);
    // n_t = tanh(i_n + r_t ⊙ h_n)
    auto new_gate = torch::tanh(chunked_igates[2] + reset_gate * chunked_hgates[2]);
    
    // h_t = (h_{t-1} - n_t) ⊙ z_t + n_t
    return (hidden - new_gate) * input_gate + new_gate;
}

/**
 * @brief Single LSTM cell
 * @param input Input tensor [batch, input_size]
 * @param hx Hidden state [batch, hidden_size]
 * @param cx Cell state [batch, hidden_size]
 * @param params LSTM parameters (weight matrices are 4*hidden_size)
 * @param pre_compute_input Whether input is pre-computed
 * @return Tuple of (new_hidden, new_cell) [batch, hidden_size]
 */
inline std::tuple<torch::Tensor, torch::Tensor> lstm_cell(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const RNNParams& params,
    bool pre_compute_input = false) {
    
    auto ih_part = pre_compute_input ? input : 
        torch::linear(input, params.weight_ih, params.bias_ih);
    auto hh_part = torch::linear(hx, params.weight_hh, params.bias_hh);
    auto gates = ih_part + hh_part;
    
    auto chunked_gates = gates.chunk(4, 1);
    auto ingate = torch::sigmoid(chunked_gates[0]);     // i_t
    auto forgetgate = torch::sigmoid(chunked_gates[1]); // f_t
    auto cellgate = torch::tanh(chunked_gates[2]);      // g_t
    auto outgate = torch::sigmoid(chunked_gates[3]);    // o_t
    
    // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    auto cy = forgetgate * cx + ingate * cellgate;
    // h_t = o_t ⊙ tanh(c_t)
    auto hy = outgate * torch::tanh(cy);
    
    return std::make_tuple(hy, cy);
}

// ============================================================================
// LAYER PROCESSING FUNCTIONS
// ============================================================================

/**
 * @brief Process a single RNN layer (unidirectional)
 */
template<typename CellFunc>
inline std::tuple<torch::Tensor, torch::Tensor> process_rnn_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_hidden,  // [batch, hidden_size]
    const RNNParams& params,
    CellFunc cell_func,
    const RNNOptions& options = RNNOptions{}) {
    
    int seq_len = input.size(0);
    int batch_size = input.size(1);
    
    std::vector<torch::Tensor> outputs;
    torch::Tensor hidden = initial_hidden;
    
    // Pre-compute input transformations for CPU optimization
    torch::Tensor input_transformed;
    bool pre_compute = options.pre_compute_input && input.device().is_cpu();
    if (pre_compute) {
        input_transformed = torch::linear(input, params.weight_ih, params.bias_ih);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor x_t = pre_compute ? input_transformed[t] : input[t];
        hidden = cell_func(x_t, hidden, params, pre_compute);
        outputs.push_back(hidden);
    }
    
    return std::make_tuple(torch::stack(outputs, 0), hidden);
}

/**
 * @brief Process a single LSTM layer (unidirectional)
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> process_lstm_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_h,       // [batch, hidden_size]
    const torch::Tensor& initial_c,       // [batch, hidden_size]
    const RNNParams& params,
    const RNNOptions& options = RNNOptions{}) {
    
    int seq_len = input.size(0);
    
    std::vector<torch::Tensor> outputs;
    torch::Tensor h = initial_h;
    torch::Tensor c = initial_c;
    
    // Pre-compute input transformations for CPU optimization
    torch::Tensor input_transformed;
    bool pre_compute = options.pre_compute_input && input.device().is_cpu();
    if (pre_compute) {
        input_transformed = torch::linear(input, params.weight_ih, params.bias_ih);
    }
    
    for (int t = 0; t < seq_len; ++t) {
        torch::Tensor x_t = pre_compute ? input_transformed[t] : input[t];
        std::tie(h, c) = lstm_cell(x_t, h, c, params, pre_compute);
        outputs.push_back(h);
    }
    
    return std::make_tuple(torch::stack(outputs, 0), h, c);
}

/**
 * @brief Process bidirectional RNN layer
 */
template<typename CellFunc>
inline std::tuple<torch::Tensor, torch::Tensor> process_bidirectional_rnn_layer(
    const torch::Tensor& input,           // [seq_len, batch, input_size]
    const torch::Tensor& initial_hidden,  // [2, batch, hidden_size]
    const RNNParams& forward_params,
    const RNNParams& backward_params,
    CellFunc cell_func,
    const RNNOptions& options = RNNOptions{}) {
    
    int seq_len = input.size(0);
    int batch_size = input.size(1);
    
    // Forward direction
    auto [fw_output, fw_hidden] = process_rnn_layer(
        input, initial_hidden[0], forward_params, cell_func, options);
    
    // Backward direction - reverse input sequence
    auto reversed_input = torch::flip(input, {0});
    auto [bw_output_rev, bw_hidden] = process_rnn_layer(
        reversed_input, initial_hidden[1], backward_params, cell_func, options);
    
    // Reverse backward output to match forward direction
    auto bw_output = torch::flip(bw_output_rev, {0});
    
    // Concatenate forward and backward outputs
    auto combined_output = torch::cat({fw_output, bw_output}, 2);
    auto combined_hidden = torch::stack({fw_hidden, bw_hidden}, 0);
    
    return std::make_tuple(combined_output, combined_hidden);
}

// ============================================================================
// HIGH-LEVEL FUNCTIONAL API
// ============================================================================

/**
 * @brief RNN with Tanh activation (functional interface)
 * @param input Input tensor [seq_len, batch, input_size] or [batch, seq_len, input_size]
 * @param hx Initial hidden state [num_layers * num_directions, batch, hidden_size]
 * @param params Vector of RNN parameters for each layer/direction
 * @param options RNN options and configuration
 * @return Tuple of (output, final_hidden)
 */
inline std::tuple<torch::Tensor, torch::Tensor> rnn_tanh(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<RNNParams>& params,
    const RNNOptions& options = RNNOptions{}) {
    
    // Handle batch_first format
    torch::Tensor x = options.batch_first ? input.transpose(0, 1) : input;
    
    int num_directions = options.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_outputs;
    
    for (int layer = 0; layer < options.num_layers; ++layer) {
        torch::Tensor layer_hidden = hx.select(0, layer * num_directions);
        
        if (options.bidirectional) {
            torch::Tensor layer_hidden_bi = hx.narrow(0, layer * num_directions, 2);
            int fw_idx = layer * num_directions;
            int bw_idx = layer * num_directions + 1;
            
            auto [layer_output, layer_final_hidden] = process_bidirectional_rnn_layer(
                x, layer_hidden_bi, params[fw_idx], params[bw_idx], rnn_tanh_cell, options);
            
            layer_outputs.push_back(layer_final_hidden);
            x = layer_output;
        } else {
            int idx = layer * num_directions;
            auto [layer_output, layer_final_hidden] = process_rnn_layer(
                x, layer_hidden, params[idx], rnn_tanh_cell, options);
            
            layer_outputs.push_back(layer_final_hidden.unsqueeze(0));
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (options.dropout > 0.0 && options.training && layer < options.num_layers - 1) {
            x = torch::dropout(x, options.dropout, options.training);
        }
    }
    
    torch::Tensor final_hidden = torch::cat(layer_outputs, 0);
    torch::Tensor output = options.batch_first ? x.transpose(0, 1) : x;
    
    return std::make_tuple(output, final_hidden);
}

/**
 * @brief RNN with ReLU activation (functional interface)
 */
inline std::tuple<torch::Tensor, torch::Tensor> rnn_relu(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<RNNParams>& params,
    const RNNOptions& options = RNNOptions{}) {
    
    // Handle batch_first format
    torch::Tensor x = options.batch_first ? input.transpose(0, 1) : input;
    
    int num_directions = options.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_outputs;
    
    for (int layer = 0; layer < options.num_layers; ++layer) {
        torch::Tensor layer_hidden = hx.select(0, layer * num_directions);
        
        if (options.bidirectional) {
            torch::Tensor layer_hidden_bi = hx.narrow(0, layer * num_directions, 2);
            int fw_idx = layer * num_directions;
            int bw_idx = layer * num_directions + 1;
            
            auto [layer_output, layer_final_hidden] = process_bidirectional_rnn_layer(
                x, layer_hidden_bi, params[fw_idx], params[bw_idx], rnn_relu_cell, options);
            
            layer_outputs.push_back(layer_final_hidden);
            x = layer_output;
        } else {
            int idx = layer * num_directions;
            auto [layer_output, layer_final_hidden] = process_rnn_layer(
                x, layer_hidden, params[idx], rnn_relu_cell, options);
            
            layer_outputs.push_back(layer_final_hidden.unsqueeze(0));
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (options.dropout > 0.0 && options.training && layer < options.num_layers - 1) {
            x = torch::dropout(x, options.dropout, options.training);
        }
    }
    
    torch::Tensor final_hidden = torch::cat(layer_outputs, 0);
    torch::Tensor output = options.batch_first ? x.transpose(0, 1) : x;
    
    return std::make_tuple(output, final_hidden);
}

/**
 * @brief GRU (functional interface)
 * @param input Input tensor [seq_len, batch, input_size] or [batch, seq_len, input_size]
 * @param hx Initial hidden state [num_layers * num_directions, batch, hidden_size]
 * @param params Vector of GRU parameters for each layer/direction
 * @param options RNN options and configuration
 * @return Tuple of (output, final_hidden)
 */
inline std::tuple<torch::Tensor, torch::Tensor> gru(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<RNNParams>& params,
    const RNNOptions& options = RNNOptions{}) {
    
    // Handle batch_first format
    torch::Tensor x = options.batch_first ? input.transpose(0, 1) : input;
    
    int num_directions = options.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_outputs;
    
    for (int layer = 0; layer < options.num_layers; ++layer) {
        torch::Tensor layer_hidden = hx.select(0, layer * num_directions);
        
        if (options.bidirectional) {
            torch::Tensor layer_hidden_bi = hx.narrow(0, layer * num_directions, 2);
            int fw_idx = layer * num_directions;
            int bw_idx = layer * num_directions + 1;
            
            auto [layer_output, layer_final_hidden] = process_bidirectional_rnn_layer(
                x, layer_hidden_bi, params[fw_idx], params[bw_idx], gru_cell, options);
            
            layer_outputs.push_back(layer_final_hidden);
            x = layer_output;
        } else {
            int idx = layer * num_directions;
            auto [layer_output, layer_final_hidden] = process_rnn_layer(
                x, layer_hidden, params[idx], gru_cell, options);
            
            layer_outputs.push_back(layer_final_hidden.unsqueeze(0));
            x = layer_output;
        }
        
        // Apply dropout between layers (except last layer)
        if (options.dropout > 0.0 && options.training && layer < options.num_layers - 1) {
            x = torch::dropout(x, options.dropout, options.training);
        }
    }
    
    torch::Tensor final_hidden = torch::cat(layer_outputs, 0);
    torch::Tensor output = options.batch_first ? x.transpose(0, 1) : x;
    
    return std::make_tuple(output, final_hidden);
}

/**
 * @brief LSTM (functional interface)
 * @param input Input tensor [seq_len, batch, input_size] or [batch, seq_len, input_size]
 * @param hx Initial hidden state [num_layers * num_directions, batch, hidden_size]
 * @param cx Initial cell state [num_layers * num_directions, batch, hidden_size]
 * @param params Vector of LSTM parameters for each layer/direction
 * @param options RNN options and configuration
 * @return Tuple of (output, final_hidden, final_cell)
 */
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstm(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const std::vector<RNNParams>& params,
    const RNNOptions& options = RNNOptions{}) {
    
    // Handle batch_first format
    torch::Tensor x = options.batch_first ? input.transpose(0, 1) : input;
    
    int num_directions = options.bidirectional ? 2 : 1;
    std::vector<torch::Tensor> layer_h_outputs, layer_c_outputs;
    
    for (int layer = 0; layer < options.num_layers; ++layer) {
        if (options.bidirectional) {
            // Bidirectional LSTM processing
            int fw_idx = layer * num_directions;
            int bw_idx = layer * num_directions + 1;
            
            torch::Tensor fw_h = hx.select(0, fw_idx);
            torch::Tensor fw_c = cx.select(0, fw_idx);
            torch::Tensor bw_h = hx.select(0, bw_idx);
            torch::Tensor bw_c = cx.select(0, bw_idx);
            
            // Forward direction
            auto [fw_output, fw_final_h, fw_final_c] = process_lstm_layer(
                x, fw_h, fw_c, params[fw_idx], options);
            
            // Backward direction
            auto reversed_input = torch::flip(x, {0});
            auto [bw_output_rev, bw_final_h, bw_final_c] = process_lstm_layer(
                reversed_input, bw_h, bw_c, params[bw_idx], options);
            auto bw_output = torch::flip(bw_output_rev, {0});
            
            // Combine outputs
            x = torch::cat({fw_output, bw_output}, 2);
            layer_h_outputs.push_back(torch::stack({fw_final_h, bw_final_h}, 0));
            layer_c_outputs.push_back(torch::stack({fw_final_c, bw_final_c}, 0));
        } else {
            // Unidirectional LSTM processing
            int idx = layer * num_directions;
            torch::Tensor layer_h = hx.select(0, idx);
            torch::Tensor layer_c = cx.select(0, idx);
            
            auto [layer_output, layer_final_h, layer_final_c] = process_lstm_layer(
                x, layer_h, layer_c, params[idx], options);
            
            layer_h_outputs.push_back(layer_final_h.unsqueeze(0));
            layer_c_outputs.push_back(layer_final_c.unsqueeze(0));
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
    
    return std::make_tuple(output, final_h, final_c);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Create RNN parameters with proper initialization
 */
inline RNNParams create_rnn_params(int input_size, int hidden_size, int gate_multiplier = 1, 
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
    
    return RNNParams(w_ih, w_hh, b_ih, b_hh);
}

/**
 * @brief Create parameter vector for multi-layer RNN
 */
inline std::vector<RNNParams> create_rnn_params_vector(
    int input_size, int hidden_size, int num_layers, bool bidirectional,
    int gate_multiplier = 1, bool use_bias = true, torch::ScalarType dtype = torch::kFloat32) {
    
    std::vector<RNNParams> params;
    int num_directions = bidirectional ? 2 : 1;
    
    for (int layer = 0; layer < num_layers; ++layer) {
        int layer_input_size = (layer == 0) ? input_size : hidden_size * num_directions;
        
        for (int direction = 0; direction < num_directions; ++direction) {
            params.push_back(create_rnn_params(layer_input_size, hidden_size, 
                                             gate_multiplier, use_bias, dtype));
        }
    }
    
    return params;
}

/**
 * @brief Create initial hidden state
 */
inline torch::Tensor create_initial_hidden(int num_layers, bool bidirectional, 
                                          int batch_size, int hidden_size,
                                          const torch::TensorOptions& options = torch::kFloat32) {
    int num_directions = bidirectional ? 2 : 1;
    return torch::zeros({num_layers * num_directions, batch_size, hidden_size}, options);
}

/**
 * @brief Simplified interface matching PyTorch's functional API exactly
 */
} // namespace custom


namespace made {

// Simplified RNN Tanh
inline std::tuple<torch::Tensor, torch::Tensor> rnn_tanh(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<custom::RNNParams>& params,
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
    
    return custom::rnn_tanh(input, hx, params, options);
}

// Simplified RNN ReLU
inline std::tuple<torch::Tensor, torch::Tensor> rnn_relu(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<custom::RNNParams>& params,
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
    
    return custom::rnn_relu(input, hx, params, options);
}

// Simplified GRU
inline std::tuple<torch::Tensor, torch::Tensor> gru(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const std::vector<custom::RNNParams>& params,
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
    
    return custom::gru(input, hx, params, options);
}

// Simplified LSTM
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstm(
    const torch::Tensor& input,
    const torch::Tensor& hx,
    const torch::Tensor& cx,
    const std::vector<custom::RNNParams>& params,
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
    
    return custom::lstm(input, hx, cx, params, options);
}

} // namespace simple