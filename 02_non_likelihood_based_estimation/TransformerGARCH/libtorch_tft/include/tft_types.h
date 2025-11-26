#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace tft {

// Data types enum (equivalent to Python DataTypes)
enum class DataType {
    REAL_VALUED = 0,
    CATEGORICAL = 1,
    DATE = 2
};

// Input types enum (equivalent to Python InputTypes)
enum class InputType {
    TARGET = 0,
    OBSERVED_INPUT = 1,
    KNOWN_INPUT = 2,
    STATIC_INPUT = 3,
    ID = 4,
    TIME = 5
};

// Column definition tuple
struct ColumnDefinition {
    std::string name;
    DataType data_type;
    InputType input_type;
    
    ColumnDefinition(const std::string& n, DataType dt, InputType it)
        : name(n), data_type(dt), input_type(it) {}
};

// TFT Configuration structure
struct TFTConfig {
    // Data parameters
    int total_time_steps;
    int num_encoder_steps;
    int input_size;
    int output_size;
    std::vector<int> category_counts;
    
    // Model parameters
    int hidden_layer_size;
    float dropout_rate;
    int num_heads;
    int num_stacks;
    std::vector<float> quantiles;
    
    // Training parameters
    float learning_rate;
    float max_gradient_norm;
    int batch_size;
    int num_epochs;
    int early_stopping_patience;
    
    // Input indices
    std::vector<int> input_obs_loc;
    std::vector<int> static_input_loc;
    std::vector<int> known_regular_input_idx;
    std::vector<int> known_categorical_input_idx;
    
    // Column definitions
    std::vector<ColumnDefinition> column_definitions;
    
    // Default constructor with reasonable defaults
    TFTConfig() : 
        total_time_steps(192),
        num_encoder_steps(168), 
        input_size(1),
        output_size(1),
        hidden_layer_size(160),
        dropout_rate(0.1f),
        num_heads(4),
        num_stacks(1),
        quantiles({0.1f, 0.5f, 0.9f}),
        learning_rate(1e-3f),
        max_gradient_norm(1.0f),
        batch_size(64),
        num_epochs(100),
        early_stopping_patience(10) {}
};

// Training data structure
struct TFTData {
    torch::Tensor inputs;
    torch::Tensor outputs;
    torch::Tensor active_entries;
    torch::Tensor time;
    torch::Tensor identifiers;
    
    TFTData() = default;
    
    TFTData(const torch::Tensor& inp, const torch::Tensor& out, 
            const torch::Tensor& active, const torch::Tensor& t, 
            const torch::Tensor& ids)
        : inputs(inp), outputs(out), active_entries(active), time(t), identifiers(ids) {}
    
    // Move to device
    TFTData to(torch::Device device) const {
        return TFTData(
            inputs.to(device),
            outputs.to(device), 
            active_entries.to(device),
            time.to(device),
            identifiers.to(device)
        );
    }
    
    // Get batch size
    int64_t batch_size() const {
        return inputs.size(0);
    }
    
    // Get sequence length
    int64_t sequence_length() const {
        return inputs.size(1);
    }
};

// Attention weights for interpretability
struct AttentionWeights {
    torch::Tensor decoder_self_attn;
    torch::Tensor static_flags;
    torch::Tensor historical_flags;
    torch::Tensor future_flags;
    
    AttentionWeights() = default;
    
    AttentionWeights(const torch::Tensor& self_attn, const torch::Tensor& static_f,
                    const torch::Tensor& hist_f, const torch::Tensor& fut_f)
        : decoder_self_attn(self_attn), static_flags(static_f), 
          historical_flags(hist_f), future_flags(fut_f) {}
};

// Prediction results
struct TFTPredictions {
    torch::Tensor predictions;  // [batch_size, forecast_steps, output_size * num_quantiles]
    AttentionWeights attention_weights;
    torch::Tensor forecast_time;
    torch::Tensor identifiers;
    
    TFTPredictions() = default;
    
    TFTPredictions(const torch::Tensor& pred, const AttentionWeights& att_weights,
                   const torch::Tensor& f_time, const torch::Tensor& ids)
        : predictions(pred), attention_weights(att_weights), 
          forecast_time(f_time), identifiers(ids) {}
};

} // namespace tft
