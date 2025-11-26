#include "tft_model.h"
#include "tft_types.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <iomanip>

using namespace tft;

// Helper function to print prediction results
void print_predictions(const TFTPredictions& predictions, const std::vector<float>& quantiles) {
    auto pred_data = predictions.predictions;
    auto batch_size = pred_data.size(0);
    auto forecast_steps = pred_data.size(1);
    auto output_size = pred_data.size(2) / quantiles.size();
    
    std::cout << "\n=== PREDICTION RESULTS ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Forecast steps: " << forecast_steps << std::endl;
    std::cout << "Output size: " << output_size << std::endl;
    std::cout << "Quantiles: ";
    for (const auto& q : quantiles) {
        std::cout << q << " ";
    }
    std::cout << "\n" << std::endl;
    
    // Print first sample predictions
    std::cout << "Sample 0 predictions:" << std::endl;
    std::cout << std::setw(8) << "Step";
    for (size_t q = 0; q < quantiles.size(); ++q) {
        std::cout << std::setw(12) << ("Q" + std::to_string(int(quantiles[q] * 100)));
    }
    std::cout << std::endl;
    
    for (int t = 0; t < std::min(10, static_cast<int>(forecast_steps)); ++t) {
        std::cout << std::setw(8) << (t + 1);
        for (size_t q = 0; q < quantiles.size(); ++q) {
            int idx = q * output_size;
            float value = pred_data[0][t][idx].item<float>();
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << value;
        }
        std::cout << std::endl;
    }
}

// Helper function to print attention weights
void print_attention_weights(const AttentionWeights& attention_weights) {
    std::cout << "\n=== ATTENTION ANALYSIS ===" << std::endl;
    
    if (attention_weights.static_flags.defined()) {
        std::cout << "Static variable importance:" << std::endl;
        auto static_weights = attention_weights.static_flags;
        for (int i = 0; i < static_weights.size(1); ++i) {
            float weight = static_weights[0][i].item<float>();
            std::cout << "  Variable " << i << ": " << std::fixed << std::setprecision(4) << weight << std::endl;
        }
    }
    
    if (attention_weights.historical_flags.defined()) {
        std::cout << "\nHistorical variables average importance:" << std::endl;
        auto hist_weights = torch::mean(attention_weights.historical_flags, 1);  // Average over time
        for (int i = 0; i < hist_weights.size(1); ++i) {
            float weight = hist_weights[0][i].item<float>();
            std::cout << "  Variable " << i << ": " << std::fixed << std::setprecision(4) << weight << std::endl;
        }
    }
    
    if (attention_weights.future_flags.defined()) {
        std::cout << "\nFuture variables average importance:" << std::endl;
        auto future_weights = torch::mean(attention_weights.future_flags, 1);  // Average over time
        for (int i = 0; i < future_weights.size(1); ++i) {
            float weight = future_weights[0][i].item<float>();
            std::cout << "  Variable " << i << ": " << std::fixed << std::setprecision(4) << weight << std::endl;
        }
    }
    
    if (attention_weights.decoder_self_attn.defined()) {
        std::cout << "\nSelf-attention pattern (first head, first 5x5 positions):" << std::endl;
        auto self_attn = attention_weights.decoder_self_attn;
        for (int i = 0; i < std::min(5, static_cast<int>(self_attn.size(-2))); ++i) {
            std::cout << "  ";
            for (int j = 0; j < std::min(5, static_cast<int>(self_attn.size(-1))); ++j) {
                float weight = self_attn[0][0][i][j].item<float>();  // [batch, head, seq, seq]
                std::cout << std::fixed << std::setprecision(3) << weight << " ";
            }
            std::cout << std::endl;
        }
    }
}

int main() {
    try {
        std::cout << "=== LibTorch TFT Prediction Example ===" << std::endl;
        
        // Check if CUDA is available
        torch::Device device(torch::kCPU);
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA, 0);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }
        
        // Configure TFT (same as training)
        TFTConfig config;
        config.total_time_steps = 192;
        config.num_encoder_steps = 168;
        config.input_size = 5;
        config.output_size = 1;
        config.hidden_layer_size = 160;
        config.dropout_rate = 0.1f;
        config.num_heads = 4;
        config.num_stacks = 1;
        config.quantiles = {0.1f, 0.5f, 0.9f};
        
        config.category_counts = {};
        config.input_obs_loc = {0};
        config.static_input_loc = {};
        config.known_regular_input_idx = {1, 2, 3, 4};
        config.known_categorical_input_idx = {};
        
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
        
        // Try to load trained model
        std::cout << "Loading trained model..." << std::endl;
        try {
            model->load("final_tft_model.pt");
            std::cout << "Loaded trained model successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Could not load trained model (" << e.what() << "), using random weights for demonstration." << std::endl;
        }
        
        // Generate some test data
        std::cout << "Generating test data..." << std::endl;
        auto test_inputs = torch::randn({3, config.total_time_steps, config.input_size}).to(device);
        
        std::cout << "Test data shape: " << test_inputs.sizes() << std::endl;
        
        // Make predictions
        std::cout << "Making predictions..." << std::endl;
        auto predictions = model->predict(test_inputs);
        
        // Print results
        print_predictions(predictions, config.quantiles);
        
        // Print attention weights for interpretability
        print_attention_weights(predictions.attention_weights);
        
        // Demonstrate extracting specific quantile predictions
        std::cout << "\n=== EXTRACTING SPECIFIC QUANTILES ===" << std::endl;
        auto all_predictions = predictions.predictions;
        auto output_size = config.output_size;
        
        for (size_t q_idx = 0; q_idx < config.quantiles.size(); ++q_idx) {
            auto quantile_pred = all_predictions.narrow(2, q_idx * output_size, output_size);
            std::cout << "Quantile " << config.quantiles[q_idx] << " predictions shape: " 
                      << quantile_pred.sizes() << std::endl;
            
            // Show mean prediction for this quantile
            auto mean_pred = torch::mean(quantile_pred);
            std::cout << "  Mean prediction: " << mean_pred.item<float>() << std::endl;
        }
        
        // Demonstrate batch prediction processing
        std::cout << "\n=== BATCH PROCESSING EXAMPLE ===" << std::endl;
        std::cout << "Processing " << predictions.predictions.size(0) << " samples in batch" << std::endl;
        
        for (int batch_idx = 0; batch_idx < predictions.predictions.size(0); ++batch_idx) {
            auto sample_pred = predictions.predictions[batch_idx];
            auto median_forecast = sample_pred.narrow(1, 1 * output_size, output_size);  // Middle quantile (0.5)
            auto mean_forecast = torch::mean(median_forecast);
            
            std::cout << "Sample " << batch_idx << " - Mean median forecast: " 
                      << std::fixed << std::setprecision(4) << mean_forecast.item<float>() << std::endl;
        }
        
        std::cout << "\nPrediction completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
