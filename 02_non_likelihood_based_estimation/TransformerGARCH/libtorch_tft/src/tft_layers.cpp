#include "tft_layers.h"
#include <torch/torch.h>
#include <stdexcept>
#include <cmath>

namespace tft {

// Helper function for applying activations
torch::Tensor apply_activation(torch::Tensor x, const std::string& activation) {
    if (activation.empty() || activation == "none") {
        return x;
    } else if (activation == "relu") {
        return torch::relu(x);
    } else if (activation == "tanh") {
        return torch::tanh(x);
    } else if (activation == "sigmoid") {
        return torch::sigmoid(x);
    } else if (activation == "elu") {
        return torch::elu(x);
    } else if (activation == "softmax") {
        return torch::softmax(x, -1);
    } else {
        throw std::runtime_error("Unknown activation: " + activation);
    }
}

// Get decoder mask for causal self-attention
torch::Tensor get_decoder_mask(torch::Tensor inputs) {
    auto len_s = inputs.size(1);
    auto batch_size = inputs.size(0);
    auto mask = torch::triu(torch::ones({len_s, len_s}), 1);
    mask = mask.unsqueeze(0).expand({batch_size, len_s, len_s});
    return mask;
}

// LinearLayerImpl
LinearLayerImpl::LinearLayerImpl()
    : linear_(nullptr),
      use_time_distributed_(false),
      activation_("") {
}

LinearLayerImpl::LinearLayerImpl(int input_size, int output_size, 
                                bool use_time_distributed, bool use_bias,
                                const std::string& activation)
    : linear_(register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(input_size, output_size).bias(use_bias)))),
      use_time_distributed_(use_time_distributed),
      activation_(activation) {
}

torch::Tensor LinearLayerImpl::forward(torch::Tensor x) {
    torch::Tensor out;
    
    if (use_time_distributed_) {
        // Apply linear layer across time dimension
        auto original_shape = x.sizes();
        auto batch_size = original_shape[0];
        auto seq_len = original_shape[1];
        auto feature_dim = original_shape[2];
        
        // Reshape to [batch_size * seq_len, features]
        x = x.view({batch_size * seq_len, feature_dim});
        out = linear_(x);
        
        // Restore temporal dimension
        auto output_dim = out.size(1);
        out = out.view({batch_size, seq_len, output_dim});
    } else {
        out = linear_(x);
    }
    
    return apply_activation(out, activation_);
}

// MLPImpl
MLPImpl::MLPImpl(int input_size, int hidden_size, int output_size,
                const std::string& hidden_activation,
                const std::string& output_activation,
                bool use_time_distributed)
    : hidden_layer_(register_module("hidden", LinearLayer(input_size, hidden_size, use_time_distributed, true, hidden_activation))),
      output_layer_(register_module("output", LinearLayer(hidden_size, output_size, use_time_distributed, true, output_activation))) {
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    x = hidden_layer_(x);
    return output_layer_(x);
}

// GatedLinearUnitImpl
GatedLinearUnitImpl::GatedLinearUnitImpl(int input_size, int hidden_size,
                                        float dropout_rate,
                                        bool use_time_distributed,
                                        const std::string& activation)
    : dropout_(register_module("dropout", torch::nn::Dropout(dropout_rate))),
      activation_layer_(register_module("activation", LinearLayer(input_size, hidden_size, use_time_distributed, true, activation))),
      gated_layer_(register_module("gated", LinearLayer(input_size, hidden_size, use_time_distributed, true, "sigmoid"))),
      dropout_rate_(dropout_rate) {
}

std::pair<torch::Tensor, torch::Tensor> GatedLinearUnitImpl::forward(torch::Tensor x) {
    if (dropout_rate_ > 0.0f) {
        x = dropout_(x);
    }
    
    auto activation_output = activation_layer_(x);
    auto gate = gated_layer_(x);
    auto output = activation_output * gate;
    
    return std::make_pair(output, gate);
}

// AddAndNormImpl
AddAndNormImpl::AddAndNormImpl(int normalized_shape)
    : layer_norm_(register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({normalized_shape})))) {
}

torch::Tensor AddAndNormImpl::forward(const std::vector<torch::Tensor>& inputs) {
    torch::Tensor sum = inputs[0];
    for (size_t i = 1; i < inputs.size(); ++i) {
        sum = sum + inputs[i];
    }
    return layer_norm_(sum);
}

// GatedResidualNetworkImpl
GatedResidualNetworkImpl::GatedResidualNetworkImpl(int input_size, int hidden_size,
                                                  int output_size, float dropout_rate,
                                                  bool use_time_distributed, bool return_gate)
    : fc1_(register_module("fc1", LinearLayer(input_size, hidden_size, use_time_distributed, true, ""))),
      fc2_(register_module("fc2", LinearLayer(hidden_size, hidden_size, use_time_distributed, true, ""))),
      glu_(register_module("glu", GatedLinearUnit(hidden_size, output_size == -1 ? input_size : output_size, dropout_rate, use_time_distributed, ""))),
      add_and_norm_(register_module("add_norm", AddAndNorm(output_size == -1 ? input_size : output_size))),
      output_size_(output_size == -1 ? input_size : output_size),
      has_skip_(output_size == -1 || output_size == input_size),
      return_gate_(return_gate) {
    
    if (!has_skip_) {
        skip_layer_ = register_module("skip", LinearLayer(input_size, output_size_, use_time_distributed, true, ""));
    }
}

std::pair<torch::Tensor, torch::Tensor> GatedResidualNetworkImpl::forward(torch::Tensor x, torch::Tensor additional_context) {
    torch::Tensor skip = has_skip_ ? x : skip_layer_(x);
    
    torch::Tensor hidden = fc1_(x);
    if (additional_context.defined()) {
        auto context_layer = LinearLayer(additional_context.size(-1), hidden.size(-1), /*use_time_distributed=*/hidden.dim() == 3, /*use_bias=*/false, "");
        hidden = hidden + context_layer(additional_context);
    }
    
    hidden = apply_activation(hidden, "elu");
    hidden = fc2_(hidden);
    
    auto [gated_output, gate] = glu_(hidden);
    std::vector<torch::Tensor> inputs_for_norm = {skip, gated_output};
    auto output = add_and_norm_(inputs_for_norm);
    
    if (return_gate_) {
        return std::make_pair(output, gate);
    } else {
        return std::make_pair(output, torch::Tensor{});
    }
}

// ScaledDotProductAttentionImpl
ScaledDotProductAttentionImpl::ScaledDotProductAttentionImpl(float dropout_rate)
    : dropout_(register_module("dropout", torch::nn::Dropout(dropout_rate))) {
}

std::pair<torch::Tensor, torch::Tensor> ScaledDotProductAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {
    auto d_k = k.size(-1);
    auto temper = std::sqrt(static_cast<float>(d_k));
    
    // Compute attention scores
    auto attn = torch::matmul(q, k.transpose(-2, -1)) / temper;
    
    // Apply mask if provided
    if (mask.defined()) {
        attn = attn.masked_fill(mask.to(torch::kBool), -1e9);
    }
    
    // Apply softmax
    attn = torch::softmax(attn, -1);
    attn = dropout_(attn);
    
    // Apply attention to values
    auto output = torch::matmul(attn, v);
    
    return std::make_pair(output, attn);
}

// InterpretableMultiHeadAttentionImpl
InterpretableMultiHeadAttentionImpl::InterpretableMultiHeadAttentionImpl(int num_heads, int d_model, float dropout_rate)
    : num_heads_(num_heads),
      d_k_(d_model / num_heads),
      d_v_(d_model / num_heads),
      dropout_rate_(dropout_rate),
      attention_(register_module("attention", ScaledDotProductAttention(dropout_rate))),
      w_o_(register_module("w_o", LinearLayer(d_model, d_model, false, false, ""))) {
    
    // Create query, key, and value layers for each head
    for (int i = 0; i < num_heads; ++i) {
        qs_layers_.push_back(register_module("qs_" + std::to_string(i), LinearLayer(d_model, d_k_, false, false, "")));
        ks_layers_.push_back(register_module("ks_" + std::to_string(i), LinearLayer(d_model, d_k_, false, false, "")));
    }
    
    // Use same value layer for all heads for interpretability
    auto vs_layer = register_module("vs", LinearLayer(d_model, d_v_, false, false, ""));
    for (int i = 0; i < num_heads; ++i) {
        vs_layers_.push_back(vs_layer);
    }
}

std::pair<torch::Tensor, torch::Tensor> InterpretableMultiHeadAttentionImpl::forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor mask) {
    std::vector<torch::Tensor> heads;
    std::vector<torch::Tensor> attns;
    
    for (int i = 0; i < num_heads_; ++i) {
        auto qs = qs_layers_[i](q);
        auto ks = ks_layers_[i](k);
        auto vs = vs_layers_[i](v);
        
        auto [head, attn] = attention_(qs, ks, vs, mask);
        head = torch::dropout(head, dropout_rate_, is_training());
        
        heads.push_back(head);
        attns.push_back(attn);
    }
    
    torch::Tensor output;
    torch::Tensor attn_weights;
    
    if (num_heads_ > 1) {
        output = torch::mean(torch::stack(heads, 0), 0);
        attn_weights = torch::stack(attns, 0);
    } else {
        output = heads[0];
        attn_weights = attns[0];
    }
    
    output = w_o_(output);
    output = torch::dropout(output, dropout_rate_, is_training());
    
    return std::make_pair(output, attn_weights);
}

// VariableSelectionNetworkImpl
VariableSelectionNetworkImpl::VariableSelectionNetworkImpl(int input_size, int num_inputs, int hidden_size,
                                                          float dropout_rate, bool use_time_distributed,
                                                          torch::Tensor additional_context)
    : num_inputs_(num_inputs),
      selection_weights_grn_(register_module("selection_grn", GatedResidualNetwork(input_size * num_inputs, hidden_size, num_inputs, dropout_rate, use_time_distributed, false))),
      use_time_distributed_(use_time_distributed) {
    
    // Create GRN for each input variable
    for (int i = 0; i < num_inputs; ++i) {
        input_grns_.push_back(register_module("input_grn_" + std::to_string(i), 
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout_rate, use_time_distributed, false)));
    }
}

std::pair<torch::Tensor, torch::Tensor> VariableSelectionNetworkImpl::forward(torch::Tensor inputs, torch::Tensor context) {
    auto original_shape = inputs.sizes();
    auto batch_size = original_shape[0];
    auto time_steps = use_time_distributed_ ? original_shape[1] : 1;
    auto embedding_dim = use_time_distributed_ ? original_shape[2] : original_shape[1];
    auto num_inputs = original_shape[-1];
    
    // Flatten inputs for selection weight computation
    torch::Tensor flatten;
    if (use_time_distributed_) {
        flatten = inputs.view({batch_size, time_steps, embedding_dim * num_inputs});
    } else {
        flatten = inputs.view({batch_size, embedding_dim * num_inputs});
    }
    
    // Compute variable selection weights
    auto [selection_weights, _] = selection_weights_grn_(flatten, context);
    selection_weights = torch::softmax(selection_weights, -1);
    
    if (use_time_distributed_) {
        selection_weights = selection_weights.unsqueeze(2);
    } else {
        selection_weights = selection_weights.unsqueeze(-1);
    }
    
    // Apply transformations to each input
    std::vector<torch::Tensor> transformed_inputs;
    for (int i = 0; i < num_inputs; ++i) {
        torch::Tensor input_i;
        if (use_time_distributed_) {
            input_i = inputs.select(-1, i);  // [batch_size, time_steps, embedding_dim]
        } else {
            input_i = inputs.select(-1, i);  // [batch_size, embedding_dim]
        }
        
        auto [transformed, _] = input_grns_[i](input_i);
        transformed_inputs.push_back(transformed);
    }
    
    auto transformed_embedding = torch::stack(transformed_inputs, -1);
    auto weighted_inputs = transformed_embedding * selection_weights;
    auto output = torch::sum(weighted_inputs, -1);
    
    return std::make_pair(output, selection_weights.squeeze(-1));
}

} // namespace tft
