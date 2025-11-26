#include "tft_model.h"
#include "tft_types.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>

using namespace tft;

// Helper function to generate synthetic time series data
TFTData generate_synthetic_data(int num_samples, int sequence_length, int num_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 1.0);
    
    auto inputs = torch::randn({num_samples, sequence_length, num_features});
    
    // Generate targets as a simple trend + noise
    auto targets = torch::zeros({num_samples, sequence_length - 168, 1}); // 168 = encoder steps
    for (int i = 0; i < num_samples; ++i) {
        for (int t = 0; t < sequence_length - 168; ++t) {
            // Simple trend with some noise
            float trend = 0.1f * (t + 168) + dis(gen) * 0.2f;
            targets[i][t][0] = trend;
        }
    }
    
    auto active_entries = torch::ones_like(targets);
    auto time_indices = torch::arange(sequence_length - 168).unsqueeze(0).expand({num_samples, -1}).unsqueeze(-1).to(torch::kFloat);
    auto identifiers = torch::arange(num_samples).unsqueeze(1).expand({-1, sequence_length - 168}).unsqueeze(-1).to(torch::kFloat);
    
    return TFTData(inputs, targets, active_entries, time_indices, identifiers);
}

int main() {
    try {
        std::cout << "=== LibTorch TFT Training Example ===" << std::endl;
        
        // Check if CUDA is available
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA, 0);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }
        
        // Configure TFT
        TFTConfig config;
        config.total_time_steps = 192;  // Total sequence length
        config.num_encoder_steps = 168; // Historical context
        config.input_size = 5;          // Number of input features
        config.output_size = 1;         // Number of output features
        config.hidden_layer_size = 160;
        config.dropout_rate = 0.1f;
        config.num_heads = 4;
        config.num_stacks = 1;
        config.learning_rate = 1e-3f;
        config.max_gradient_norm = 1.0f;
        config.batch_size = 32;
        config.num_epochs = 50;
        config.early_stopping_patience = 10;
        config.quantiles = {0.1f, 0.5f, 0.9f};
        
        // No categorical variables in this example
        config.category_counts = {};
        
        // Simple configuration - assume all inputs except first are known future inputs
        config.input_obs_loc = {0};  // First column is target
        config.static_input_loc = {};
        config.known_regular_input_idx = {1, 2, 3, 4};  // Rest are known future inputs
        config.known_categorical_input_idx = {};
        
        // Column definitions (simplified)
        config.column_definitions = {
            ColumnDefinition("target", DataType::REAL_VALUED, InputType::TARGET),
            ColumnDefinition("feature1", DataType::REAL_VALUED, InputType::KNOWN_INPUT),
            ColumnDefinition("feature2", DataType::REAL_VALUED, InputType::KNOWN_INPUT),
            ColumnDefinition("feature3", DataType::REAL_VALUED, InputType::KNOWN_INPUT),
            ColumnDefinition("feature4", DataType::REAL_VALUED, InputType::KNOWN_INPUT)
        };
        
        std::cout << "Creating TFT model..." << std::endl;
        
        // Create model
        auto model = TemporalFusionTransformer(config);
        model->to(device);
        
        std::cout << "Model created successfully!" << std::endl;
        std::cout << "Total parameters: " << torch::nn::utils::parameters_to_vector(model->parameters()).numel() << std::endl;
        
        // Generate synthetic data
        std::cout << "Generating synthetic training data..." << std::endl;
        auto train_data = generate_synthetic_data(500, config.total_time_steps, config.input_size);
        auto valid_data = generate_synthetic_data(100, config.total_time_steps, config.input_size);
        
        // Move data to device
        train_data = train_data.to(device);
        valid_data = valid_data.to(device);
        
        std::cout << "Training data shape: " << train_data.inputs.sizes() << std::endl;
        std::cout << "Training targets shape: " << train_data.outputs.sizes() << std::endl;
        
        // Create trainer
        std::cout << "Setting up trainer..." << std::endl;
        TFTTrainer trainer(model, config);
        trainer.set_early_stopping_callback(config.early_stopping_patience);
        trainer.set_model_checkpoint_callback("best_tft_model.pt");
        
        // Start training
        std::cout << "Starting training..." << std::endl;
        trainer.train(train_data, valid_data);
        
        // Final evaluation
        std::cout << "Final evaluation..." << std::endl;
        auto final_loss = trainer.evaluate(valid_data);
        std::cout << "Final validation loss: " << final_loss << std::endl;
        
        // Save final model
        std::cout << "Saving final model..." << std::endl;
        model->save("final_tft_model.pt");
        
        // Test prediction
        std::cout << "Testing prediction..." << std::endl;
        auto predictions = model->predict(valid_data.inputs.slice(0, 0, 5));  // Take first 5 samples
        std::cout << "Prediction shape: " << predictions.predictions.sizes() << std::endl;
        
        std::cout << "Training completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
