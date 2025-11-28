#include "tft_model.h"
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

namespace tft {

// QuantileLoss implementation
QuantileLoss::QuantileLoss(const std::vector<float>& quantiles) : quantiles_(quantiles) {}

torch::Tensor QuantileLoss::quantile_loss(torch::Tensor y_true, torch::Tensor y_pred, float quantile) {
    if (quantile < 0.0f || quantile > 1.0f) {
        throw std::runtime_error("Quantile must be between 0 and 1");
    }
    
    auto prediction_underflow = y_true - y_pred;
    auto q_loss = quantile * torch::clamp(prediction_underflow, 0.0) + 
                  (1.0 - quantile) * torch::clamp(-prediction_underflow, 0.0);
    
    return torch::sum(q_loss, -1);
}

torch::Tensor QuantileLoss::forward(torch::Tensor predictions, torch::Tensor targets) {
    auto output_size = targets.size(-1);
    auto num_quantiles = quantiles_.size();
    
    torch::Tensor total_loss = torch::zeros_like(targets.select(-1, 0));
    
    for (size_t i = 0; i < num_quantiles; ++i) {
        auto pred_slice = predictions.narrow(-1, i * output_size, output_size);
        auto loss = quantile_loss(targets, pred_slice, quantiles_[i]);
        total_loss += loss;
    }
    
    return total_loss;
}

// TemporalFusionTransformerImpl
TemporalFusionTransformerImpl::TemporalFusionTransformerImpl(const TFTConfig& config) 
    : config_(config) {
    
    // Initialize embeddings
    int num_categorical = config_.category_counts.size();
    int num_real = config_.input_size - num_categorical;
    
    // Categorical embeddings
    for (int i = 0; i < num_categorical; ++i) {
        categorical_embeddings_.push_back(
            register_module("cat_emb_" + std::to_string(i), 
                torch::nn::Embedding(config_.category_counts[i], config_.hidden_layer_size))
        );
    }
    
    // Real value embeddings (linear transformations)
    for (int i = 0; i < num_real; ++i) {
        real_embeddings_.push_back(
            register_module("real_emb_" + std::to_string(i),
                LinearLayer(1, config_.hidden_layer_size, true, true, ""))
        );
    }
    
    // Count total inputs after embeddings
    int total_historical_inputs = 0;
    int total_future_inputs = 0;
    int total_static_inputs = 0;
    
    // Count based on input types (simplified - assumes all are historical for now)
    total_historical_inputs = config_.input_size - static_cast<int>(config_.static_input_loc.size());
    total_future_inputs = static_cast<int>(config_.known_regular_input_idx.size()) + 
                         static_cast<int>(config_.known_categorical_input_idx.size());
    total_static_inputs = static_cast<int>(config_.static_input_loc.size());
    
    // Variable Selection Networks
    if (total_static_inputs > 0) {
        static_vsn_ = register_module("static_vsn", 
            VariableSelectionNetwork(config_.hidden_layer_size, total_static_inputs, 
                                   config_.hidden_layer_size, config_.dropout_rate, false));
    }
    
    historical_vsn_ = register_module("historical_vsn",
        VariableSelectionNetwork(config_.hidden_layer_size, total_historical_inputs,
                               config_.hidden_layer_size, config_.dropout_rate, true));
    
    future_vsn_ = register_module("future_vsn", 
        VariableSelectionNetwork(config_.hidden_layer_size, total_future_inputs,
                               config_.hidden_layer_size, config_.dropout_rate, true));
    
    // Static context networks
    static_context_variable_selection_ = register_module("static_ctx_var_sel",
        GatedResidualNetwork(config_.hidden_layer_size, config_.hidden_layer_size,
                           config_.hidden_layer_size, config_.dropout_rate, false));
    
    static_context_enrichment_ = register_module("static_ctx_enrich",
        GatedResidualNetwork(config_.hidden_layer_size, config_.hidden_layer_size,
                           config_.hidden_layer_size, config_.dropout_rate, false));
    
    static_context_state_h_ = register_module("static_ctx_h",
        GatedResidualNetwork(config_.hidden_layer_size, config_.hidden_layer_size,
                           config_.hidden_layer_size, config_.dropout_rate, false));
    
    static_context_state_c_ = register_module("static_ctx_c",
        GatedResidualNetwork(config_.hidden_layer_size, config_.hidden_layer_size,
                           config_.hidden_layer_size, config_.dropout_rate, false));
    
    // LSTM encoder-decoder
    lstm_ = register_module("lstm", torch::nn::LSTM(
        torch::nn::LSTMOptions(config_.hidden_layer_size, config_.hidden_layer_size)
            .num_layers(1)
            .batch_first(true)
            .dropout(0.0)
    ));
    // Decoder LSTM for true encoder-decoder TFT
    decoder_lstm_ = register_module("decoder_lstm", torch::nn::LSTM(
        torch::nn::LSTMOptions(config_.hidden_layer_size, config_.hidden_layer_size)
            .num_layers(1)
            .batch_first(true)
            .dropout(0.0)
    ));
    
    // Gating layers
    lstm_gate_ = register_module("lstm_gate", 
        GatedLinearUnit(config_.hidden_layer_size, config_.hidden_layer_size, config_.dropout_rate, true));
    
    temporal_add_norm_ = register_module("temporal_add_norm",
        AddAndNorm(config_.hidden_layer_size));
    
    // Static enrichment
    static_enrichment_ = register_module("static_enrichment",
        GatedResidualNetwork(config_.hidden_layer_size, config_.hidden_layer_size,
                           config_.hidden_layer_size, config_.dropout_rate, true, true));
    
    // Self-attention
    self_attention_ = register_module("self_attention",
        InterpretableMultiHeadAttention(config_.num_heads, config_.hidden_layer_size, config_.dropout_rate));
    
    attention_gate_ = register_module("attention_gate",
        GatedLinearUnit(config_.hidden_layer_size, config_.hidden_layer_size, config_.dropout_rate, true));
    
    attention_add_norm_ = register_module("attention_add_norm",
        AddAndNorm(config_.hidden_layer_size));
    
    // Final processing
    decoder_grn_ = register_module("decoder_grn",
        GatedResidualNetwork(config_.hidden_layer_size, config_.hidden_layer_size,
                           config_.hidden_layer_size, config_.dropout_rate, true));
    
    final_gate_ = register_module("final_gate",
        GatedLinearUnit(config_.hidden_layer_size, config_.hidden_layer_size, 0.0, false));
    
    final_add_norm_ = register_module("final_add_norm",
        AddAndNorm(config_.hidden_layer_size));
    
    // Output layer
    int output_dim = config_.output_size * static_cast<int>(config_.quantiles.size());
    output_layer_ = register_module("output",
        LinearLayer(config_.hidden_layer_size, output_dim, true, true, ""));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TemporalFusionTransformerImpl::get_tft_embeddings(torch::Tensor all_inputs) {
    auto batch_size = all_inputs.size(0);
    auto time_steps = all_inputs.size(1);
    
    int num_categorical = static_cast<int>(config_.category_counts.size());
    int num_real = config_.input_size - num_categorical;
    
    // Split real and categorical inputs
    auto real_inputs = all_inputs.narrow(2, 0, num_real);
    torch::Tensor categorical_inputs;
    if (num_categorical > 0) {
        categorical_inputs = all_inputs.narrow(2, num_real, num_categorical);
    }
    
    // Process embeddings
    std::vector<torch::Tensor> embedded_inputs;
    
    // Real embeddings
    for (int i = 0; i < num_real; ++i) {
        auto input_slice = real_inputs.select(2, i).unsqueeze(2);  // [batch, time, 1]
        embedded_inputs.push_back(real_embeddings_[i](input_slice));
    }
    
    // Categorical embeddings
    for (int i = 0; i < num_categorical; ++i) {
        auto input_slice = categorical_inputs.select(2, i).to(torch::kLong);
        embedded_inputs.push_back(categorical_embeddings_[i](input_slice));
    }
    
    // Stack all embeddings
    auto all_embedded = torch::stack(embedded_inputs, -1);  // [batch, time, hidden, num_inputs]
    
    // Separate into different input types (simplified version)
    torch::Tensor unknown_inputs, known_inputs, obs_inputs, static_inputs;
    
    // For simplicity, assume first input is target (observed)
    obs_inputs = all_embedded.select(-1, 0).unsqueeze(-1);  // Take first input as target
    
    // Remaining inputs are known
    if (all_embedded.size(-1) > 1) {
        known_inputs = all_embedded.narrow(-1, 1, all_embedded.size(-1) - 1);
    } else {
        known_inputs = torch::zeros_like(obs_inputs);
    }
    
    // Static inputs (take first timestep of some inputs)
    if (!config_.static_input_loc.empty()) {
        std::vector<torch::Tensor> static_list;
        for (int idx : config_.static_input_loc) {
            if (idx < all_embedded.size(-1)) {
                static_list.push_back(all_embedded.select(1, 0).select(-1, idx));
            }
        }
        if (!static_list.empty()) {
            static_inputs = torch::stack(static_list, 1);  // [batch, num_static, hidden]
        }
    }
    
    if (!static_inputs.defined()) {
        static_inputs = torch::zeros({batch_size, 1, config_.hidden_layer_size});
    }
    
    // Unknown inputs (empty for simplicity)
    unknown_inputs = torch::Tensor{};
    
    return std::make_tuple(unknown_inputs, known_inputs, obs_inputs, static_inputs);
}

std::pair<torch::Tensor, torch::Tensor>
TemporalFusionTransformerImpl::static_combine_and_mask(torch::Tensor embedding) {
    if (!static_vsn_) {
        // No static inputs: produce zero context and dummy weights
        auto batch = embedding.size(0);
        auto hidden = config_.hidden_layer_size;
        auto static_vec = torch::zeros({batch, hidden}, embedding.options());
        auto weights = torch::zeros({batch, 1}, embedding.options());
        return {static_vec, weights};
    }
    auto [static_vec, static_weights] = static_vsn_(embedding);
    return std::make_pair(static_vec, static_weights);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TemporalFusionTransformerImpl::lstm_combine_and_mask(torch::Tensor embedding, torch::Tensor context) {
    auto [selected_inputs, weights] = historical_vsn_(embedding, context);
    return std::make_tuple(selected_inputs, weights, torch::Tensor{});
}

std::pair<torch::Tensor, AttentionWeights> 
TemporalFusionTransformerImpl::forward(torch::Tensor inputs) {
    // Get embeddings
    auto [unknown_inputs, known_inputs, obs_inputs, static_inputs] = get_tft_embeddings(inputs);
    
    // Static variable selection and context
    auto [static_encoder, static_weights] = static_combine_and_mask(static_inputs);
    
    // Static contexts
    auto [static_ctx_var_sel, _1] = static_context_variable_selection_(static_encoder);
    auto [static_ctx_enrich, _2] = static_context_enrichment_(static_encoder);
    auto [static_ctx_h, _3] = static_context_state_h_(static_encoder);
    auto [static_ctx_c, _4] = static_context_state_c_(static_encoder);
    
    // Combine historical inputs
    torch::Tensor historical_inputs;
    if (unknown_inputs.defined()) {
        auto hist_unknown = unknown_inputs.narrow(1, 0, config_.num_encoder_steps);
        auto hist_known = known_inputs.narrow(1, 0, config_.num_encoder_steps);
        auto hist_obs = obs_inputs.narrow(1, 0, config_.num_encoder_steps);
        historical_inputs = torch::cat({hist_unknown, hist_known, hist_obs}, -1);
    } else {
        auto hist_known = known_inputs.narrow(1, 0, config_.num_encoder_steps);
        auto hist_obs = obs_inputs.narrow(1, 0, config_.num_encoder_steps);
        historical_inputs = torch::cat({hist_known, hist_obs}, -1);
    }
    
    // Future inputs
    auto future_inputs = known_inputs.narrow(1, config_.num_encoder_steps, 
                                           config_.total_time_steps - config_.num_encoder_steps);
    
    // Variable selection for historical and future
    auto expanded_ctx = static_ctx_var_sel.unsqueeze(1);
    auto [historical_features, historical_flags, _5] = lstm_combine_and_mask(historical_inputs, expanded_ctx);
    auto [future_features, future_flags, _6] = lstm_combine_and_mask(future_inputs, expanded_ctx);
    
    // LSTM processing
    auto initial_h = static_ctx_h.unsqueeze(0);  // Add layer dimension
    auto initial_c = static_ctx_c.unsqueeze(0);
    
    auto [history_lstm, hidden_states] = lstm_->forward(historical_features, 
                                                       std::make_tuple(initial_h, initial_c));
    torch::Tensor future_lstm;
    if (future_features.size(1) > 0) {
        auto [dec_out, dec_state] = decoder_lstm_->forward(future_features, hidden_states);
        future_lstm = dec_out;
    } else {
        // No decoder horizon: create zero-length tensor
        future_lstm = torch::zeros({history_lstm.size(0), 0, history_lstm.size(2)},
                                   history_lstm.options());
    }
    // Combine LSTM outputs
    auto lstm_outputs = torch::cat({history_lstm, future_lstm}, 1);
    
    // Apply gating
    auto [gated_lstm, _7] = lstm_gate_(lstm_outputs);
    auto input_embeddings = torch::cat({historical_features, future_features}, 1);
    std::vector<torch::Tensor> temporal_inputs = {gated_lstm, input_embeddings};
    auto temporal_features = temporal_add_norm_(temporal_inputs);
    
    // Static enrichment
    auto expanded_enrich_ctx = static_ctx_enrich.unsqueeze(1);
    auto [enriched, _8] = static_enrichment_(temporal_features, expanded_enrich_ctx);
    
    // Self-attention
    auto mask = get_decoder_mask(enriched);
    auto [attn_output, self_attn_weights] = self_attention_(enriched, enriched, enriched, mask);
    
    auto [gated_attn, _9] = attention_gate_(attn_output);
    std::vector<torch::Tensor> attention_inputs = {gated_attn, enriched};
    auto attn_output_norm = attention_add_norm_(attention_inputs);
    
    // Final processing
    auto [decoder_output, _10] = decoder_grn_(attn_output_norm);
    auto [final_gated, _11] = final_gate_(decoder_output);
    std::vector<torch::Tensor> final_inputs = {final_gated, temporal_features};
    auto final_output = final_add_norm_(final_inputs);
    
    // Output only decoder steps
    auto decoder_steps = final_output.narrow(1, config_.num_encoder_steps, 
                                            config_.total_time_steps - config_.num_encoder_steps);
    auto predictions = output_layer_(decoder_steps);
    
    // Prepare attention weights
    AttentionWeights attn_weights(
        self_attn_weights,
        static_weights,
        historical_flags,
        future_flags
    );
    
    return std::make_pair(predictions, attn_weights);
}

TFTPredictions TemporalFusionTransformerImpl::predict(torch::Tensor inputs) {
    this->eval();  // Set to evaluation mode
    
    torch::NoGradGuard no_grad;
    auto [predictions, attention_weights] = forward(inputs);
    
    // Extract time and identifiers (simplified - would need proper implementation)
    auto batch_size = inputs.size(0);
    auto forecast_steps = config_.total_time_steps - config_.num_encoder_steps;
    
    auto forecast_time = torch::arange(forecast_steps).unsqueeze(0).expand({batch_size, forecast_steps});
    auto identifiers = torch::arange(batch_size).unsqueeze(1).expand({batch_size, forecast_steps});
    
    return TFTPredictions(predictions, attention_weights, forecast_time, identifiers);
}

void TemporalFusionTransformerImpl::save(const std::string& path) {
    torch::save(shared_from_this(), path);
}

void TemporalFusionTransformerImpl::load(const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);
    torch::nn::Module::load(archive);
}

// TFTTrainer implementation
TFTTrainer::TFTTrainer(TemporalFusionTransformer model, const TFTConfig& config)
    : model_(model), config_(config), criterion_(config.quantiles), 
      optimizer_(model->parameters(), torch::optim::AdamOptions(config.learning_rate)),
      current_epoch_(0), best_loss_(std::numeric_limits<float>::max()),
      patience_counter_(0), early_stopping_enabled_(false) {
    
    // Set gradient clipping
    for (auto& param : model_->parameters()) {
        if (param.grad().defined()) {
            torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.max_gradient_norm);
        }
    }
}

torch::Tensor TFTTrainer::get_active_flags(torch::Tensor active_entries) {
    return torch::sum(active_entries, -1) > 0.0;
}

std::pair<float, float> TFTTrainer::train_epoch(const TFTData& train_data, const TFTData& valid_data) {
    model_->train();
    float train_loss = 0.0f;
    int num_batches = 0;
    
    // Training loop (simplified - would need proper batching)
    optimizer_.zero_grad();
    auto [predictions, _] = model_->forward(train_data.inputs);
    
    // Prepare targets (replicate for each quantile)
    auto targets = train_data.outputs;
    std::vector<torch::Tensor> target_list;
    for (size_t i = 0; i < config_.quantiles.size(); ++i) {
        target_list.push_back(targets);
    }
    auto replicated_targets = torch::cat(target_list, -1);
    
    auto loss = criterion_.forward(predictions, replicated_targets);
    auto active_flags = get_active_flags(train_data.active_entries);
    loss = torch::mean(loss * active_flags);
    
    loss.backward();
    
    // Gradient clipping
    torch::nn::utils::clip_grad_norm_(model_->parameters(), config_.max_gradient_norm);
    
    optimizer_.step();
    
    train_loss = loss.item<float>();
    
    // Validation
    model_->eval();
    torch::NoGradGuard no_grad;
    
    auto [val_predictions, _2] = model_->forward(valid_data.inputs);
    std::vector<torch::Tensor> val_target_list;
    for (size_t i = 0; i < config_.quantiles.size(); ++i) {
        val_target_list.push_back(valid_data.outputs);
    }
    auto val_replicated_targets = torch::cat(val_target_list, -1);
    
    auto val_loss = criterion_.forward(val_predictions, val_replicated_targets);
    auto val_active_flags = get_active_flags(valid_data.active_entries);
    val_loss = torch::mean(val_loss * val_active_flags);
    
    float validation_loss = val_loss.item<float>();
    
    return std::make_pair(train_loss, validation_loss);
}

void TFTTrainer::train(const TFTData& train_data, const TFTData& valid_data) {
    std::cout << "Starting TFT training..." << std::endl;
    
    for (int epoch = 0; epoch < config_.num_epochs; ++epoch) {
        current_epoch_ = epoch;
        
        auto [train_loss, val_loss] = train_epoch(train_data, valid_data);
        
        std::cout << "Epoch " << epoch + 1 << "/" << config_.num_epochs 
                  << " - Train loss: " << train_loss << " - Val loss: " << val_loss << std::endl;
        
        // Early stopping check
        if (early_stopping_enabled_) {
            if (val_loss < best_loss_ - early_stopping_min_delta_) {
                best_loss_ = val_loss;
                patience_counter_ = 0;
                
                // Save best model
                if (!checkpoint_path_.empty()) {
                    model_->save(checkpoint_path_);
                }
            } else {
                patience_counter_++;
                if (patience_counter_ >= early_stopping_patience_) {
                    std::cout << "Early stopping triggered after " << epoch + 1 << " epochs" << std::endl;
                    break;
                }
            }
        }
    }
    
    std::cout << "Training completed." << std::endl;
}

float TFTTrainer::evaluate(const TFTData& data) {
    model_->eval();
    torch::NoGradGuard no_grad;
    
    auto [predictions, _] = model_->forward(data.inputs);
    
    std::vector<torch::Tensor> target_list;
    for (size_t i = 0; i < config_.quantiles.size(); ++i) {
        target_list.push_back(data.outputs);
    }
    auto replicated_targets = torch::cat(target_list, -1);
    
    auto loss = criterion_.forward(predictions, replicated_targets);
    auto active_flags = get_active_flags(data.active_entries);
    loss = torch::mean(loss * active_flags);
    
    return loss.item<float>();
}

void TFTTrainer::set_early_stopping_callback(int patience, float min_delta) {
    early_stopping_enabled_ = true;
    early_stopping_patience_ = patience;
    early_stopping_min_delta_ = min_delta;
}

void TFTTrainer::set_model_checkpoint_callback(const std::string& path) {
    checkpoint_path_ = path;
}

} // namespace tft
