#pragma once

#include "tft_types.h"
#include "tft_layers.h"
#include <torch/torch.h>
#include <memory>

namespace tft {

// Main Temporal Fusion Transformer model
class TemporalFusionTransformerImpl : public torch::nn::Module {
public:
    TemporalFusionTransformerImpl(const TFTConfig& config);
    
    // Forward pass
    std::pair<torch::Tensor, AttentionWeights> forward(torch::Tensor inputs);
    
    // Prediction function
    TFTPredictions predict(torch::Tensor inputs);
    
    // Training functions
    void train_model(const TFTData& train_data, const TFTData& valid_data);
    float evaluate(const TFTData& data);
    
    // Serialization
    void save(const std::string& path);
    void load(const std::string& path);
    
    // Getters
    const TFTConfig& get_config() const { return config_; }
    
private:
    TFTConfig config_;
    
    // Embedding components
    std::vector<torch::nn::Embedding> categorical_embeddings_;
    std::vector<LinearLayer> real_embeddings_;
    
    // Variable Selection Networks
    VariableSelectionNetwork static_vsn_;
    VariableSelectionNetwork historical_vsn_;
    VariableSelectionNetwork future_vsn_;
    
    // Static context networks
    GatedResidualNetwork static_context_variable_selection_;
    GatedResidualNetwork static_context_enrichment_;
    GatedResidualNetwork static_context_state_h_;
    GatedResidualNetwork static_context_state_c_;
    
    // LSTM encoder-decoder
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::LSTM decoder_lstm_{nullptr};
    
    // Gating for skip connections
    GatedLinearUnit lstm_gate_;
    AddAndNorm temporal_add_norm_;
    
    // Static enrichment
    GatedResidualNetwork static_enrichment_;
    
    // Self-attention layer
    InterpretableMultiHeadAttention self_attention_;
    GatedLinearUnit attention_gate_;
    AddAndNorm attention_add_norm_;
    
    // Final processing
    GatedResidualNetwork decoder_grn_;
    GatedLinearUnit final_gate_;
    AddAndNorm final_add_norm_;
    
    // Output layer
    LinearLayer output_layer_;
    
    // Helper functions
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
        get_tft_embeddings(torch::Tensor all_inputs);
    
    std::pair<torch::Tensor, torch::Tensor> 
        static_combine_and_mask(torch::Tensor embedding);
        
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
        lstm_combine_and_mask(torch::Tensor embedding, torch::Tensor context);
};
TORCH_MODULE(TemporalFusionTransformer);

// Quantile loss function
class QuantileLoss : public torch::nn::Module {
public:
    QuantileLoss(const std::vector<float>& quantiles);
    
    torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets);
    
private:
    std::vector<float> quantiles_;
    
    torch::Tensor quantile_loss(torch::Tensor y_true, torch::Tensor y_pred, float quantile);
};

// Trainer class for handling training loop
class TFTTrainer {
public:
    TFTTrainer(TemporalFusionTransformer model, const TFTConfig& config);
    
    void train(const TFTData& train_data, const TFTData& valid_data);
    float evaluate(const TFTData& data);
    
    // Callbacks
    void set_early_stopping_callback(int patience, float min_delta = 1e-4);
    void set_model_checkpoint_callback(const std::string& path);
    
private:
    TemporalFusionTransformer model_;
    TFTConfig config_;
    QuantileLoss criterion_;
    torch::optim::Adam optimizer_;
    
    // Training state
    int current_epoch_;
    float best_loss_;
    int patience_counter_;
    bool early_stopping_enabled_;
    int early_stopping_patience_;
    float early_stopping_min_delta_;
    std::string checkpoint_path_;
    
    // Helper functions
    torch::Tensor get_active_flags(torch::Tensor active_entries);
    std::pair<float, float> train_epoch(const TFTData& train_data, const TFTData& valid_data);
};

} // namespace tft
