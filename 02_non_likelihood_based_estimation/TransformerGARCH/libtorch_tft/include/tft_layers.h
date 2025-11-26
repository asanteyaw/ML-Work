#pragma once

#include "tft_types.h"
#include <torch/torch.h>
#include <memory>

namespace tft {

// Linear layer with optional time distribution
class LinearLayerImpl : public torch::nn::Module {
public:
    LinearLayerImpl();  // Default constructor
    LinearLayerImpl(int input_size, int output_size, 
                   bool use_time_distributed = false, 
                   bool use_bias = true,
                   const std::string& activation = "");
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    torch::nn::Linear linear_;
    bool use_time_distributed_;
    std::string activation_;
};
TORCH_MODULE(LinearLayer);

// Multi-Layer Perceptron
class MLPImpl : public torch::nn::Module {
public:
    MLPImpl(int input_size, int hidden_size, int output_size,
           const std::string& hidden_activation = "tanh",
           const std::string& output_activation = "",
           bool use_time_distributed = false);
    
    torch::Tensor forward(torch::Tensor x);
    
private:
    LinearLayer hidden_layer_;
    LinearLayer output_layer_;
};
TORCH_MODULE(MLP);

// Gated Linear Unit (GLU)
class GatedLinearUnitImpl : public torch::nn::Module {
public:
    GatedLinearUnitImpl(int input_size, int hidden_size, 
                       float dropout_rate = 0.0f,
                       bool use_time_distributed = true,
                       const std::string& activation = "");
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);  // Returns (output, gate)
    
private:
    torch::nn::Dropout dropout_;
    LinearLayer activation_layer_;
    LinearLayer gated_layer_;
    float dropout_rate_;
};
TORCH_MODULE(GatedLinearUnit);

// Add and Norm layer
class AddAndNormImpl : public torch::nn::Module {
public:
    AddAndNormImpl(int normalized_shape);
    
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs);
    
private:
    torch::nn::LayerNorm layer_norm_;
};
TORCH_MODULE(AddAndNorm);

// Gated Residual Network (GRN)
class GatedResidualNetworkImpl : public torch::nn::Module {
public:
    GatedResidualNetworkImpl(int input_size, int hidden_size, 
                            int output_size = -1,
                            float dropout_rate = 0.0f,
                            bool use_time_distributed = true,
                            bool return_gate = false);
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, 
                                                    torch::Tensor additional_context = {});
    
private:
    LinearLayer skip_layer_{nullptr};
    LinearLayer fc1_{nullptr};
    LinearLayer fc2_{nullptr}; 
    GatedLinearUnit glu_{nullptr};
    AddAndNorm add_and_norm_{nullptr};
    int output_size_;
    bool has_skip_;
    bool return_gate_;
};
TORCH_MODULE(GatedResidualNetwork);

// Scaled Dot Product Attention
class ScaledDotProductAttentionImpl : public torch::nn::Module {
public:
    ScaledDotProductAttentionImpl(float dropout_rate = 0.0f);
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor q, torch::Tensor k, 
                                                    torch::Tensor v, torch::Tensor mask = {});
    
private:
    torch::nn::Dropout dropout_;
};
TORCH_MODULE(ScaledDotProductAttention);

// Interpretable Multi-Head Attention
class InterpretableMultiHeadAttentionImpl : public torch::nn::Module {
public:
    InterpretableMultiHeadAttentionImpl(int num_heads, int d_model, float dropout_rate = 0.0f);
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor q, torch::Tensor k, 
                                                    torch::Tensor v, torch::Tensor mask = {});
    
private:
    int num_heads_;
    int d_k_, d_v_;
    float dropout_rate_;
    
    std::vector<LinearLayer> qs_layers_;
    std::vector<LinearLayer> ks_layers_;
    std::vector<LinearLayer> vs_layers_;
    ScaledDotProductAttention attention_;
    LinearLayer w_o_;
};
TORCH_MODULE(InterpretableMultiHeadAttention);

// Variable Selection Network
class VariableSelectionNetworkImpl : public torch::nn::Module {
public:
    VariableSelectionNetworkImpl(int input_size, int num_inputs, int hidden_size,
                                float dropout_rate = 0.0f, 
                                bool use_time_distributed = false,
                                torch::Tensor additional_context = {});
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor inputs, 
                                                    torch::Tensor context = {});
    
private:
    int num_inputs_;
    GatedResidualNetwork selection_weights_grn_;
    std::vector<GatedResidualNetwork> input_grns_;
    bool use_time_distributed_;
};
TORCH_MODULE(VariableSelectionNetwork);

// Utility functions
torch::Tensor get_decoder_mask(torch::Tensor inputs);
torch::Tensor apply_activation(torch::Tensor x, const std::string& activation);

} // namespace tft
