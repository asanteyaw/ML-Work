# LibTorch Temporal Fusion Transformer (TFT)

This is a faithful C++ LibTorch implementation of the **Temporal Fusion Transformer** for interpretable multi-horizon time series forecasting, based on the original Python/TensorFlow implementation.

## Overview

The Temporal Fusion Transformer (TFT) is a hybrid attention-based architecture that combines:
- **LSTM layers** for local temporal processing
- **Self-attention layers** for learning long-term dependencies  
- **Variable Selection Networks** for feature importance
- **Gated Residual Networks** for non-linear processing
- **Quantile forecasting** for prediction intervals

This implementation faithfully converts the original TFT architecture from the paper "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" by Lim et al.

## Architecture Components

### Core Components

1. **Variable Selection Networks (VSN)**
   - Static variable selection for time-invariant features
   - Temporal variable selection for historical and future inputs
   - Provides interpretable feature importance weights

2. **Gated Residual Networks (GRN)**
   - Non-linear feature processing with gating mechanisms
   - Skip connections with layer normalization
   - Supports both time-distributed and static processing

3. **LSTM Encoder-Decoder**
   - Processes historical context (encoder) 
   - Generates future representations (decoder)
   - Uses static context for initialization

4. **Interpretable Multi-Head Attention**
   - Self-attention for long-range dependencies
   - Shared value projections for interpretability
   - Temporal attention weights for explainability

5. **Quantile Loss**
   - Supports multiple quantile predictions (e.g., 10%, 50%, 90%)
   - Enables prediction intervals and uncertainty quantification

## Project Structure

```
libtorch_tft/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/
│   ├── tft_types.h            # Data types and configurations
│   ├── tft_layers.h           # Core TFT layer implementations
│   └── tft_model.h            # Main TFT model and trainer
├── src/
│   ├── tft_layers.cpp         # Layer implementations
│   └── tft_model.cpp          # Model and trainer implementations
└── examples/
    ├── train.cpp              # Training example
    └── predict.cpp            # Prediction/inference example
```

## Key Features

- **Faithful Implementation**: Direct conversion of original Python/TensorFlow TFT
- **Full Interpretability**: Variable selection weights and attention patterns
- **Quantile Forecasting**: Multiple quantile predictions with uncertainty
- **GPU Support**: CUDA acceleration when available
- **Flexible Configuration**: Easy setup for different datasets
- **Production Ready**: Model serialization and inference capabilities

## Requirements

- **LibTorch**: PyTorch C++ API (1.12.0 or later)
- **CMake**: 3.12 or later
- **C++ Compiler**: Supporting C++17 (GCC 7+, Clang 5+, MSVC 2017+)
- **CUDA** (optional): For GPU acceleration

## Installation

### 1. Install LibTorch

Download LibTorch from https://pytorch.org/cplusplus/

```bash
# Example for Linux CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.0+cpu.zip
```

### 2. Build the Project

```bash
cd libtorch_tft
mkdir build && cd build

# Configure with your LibTorch path
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# Build
make -j4
```

### 3. Run Examples

```bash
# Training example
./train_tft

# Prediction example  
./predict_tft
```

## Usage

### Basic Training

```cpp
#include "tft_model.h"
#include "tft_types.h"

using namespace tft;

// Configure TFT
TFTConfig config;
config.total_time_steps = 192;    // Total sequence length
config.num_encoder_steps = 168;   // Historical context length
config.input_size = 5;            // Number of input features
config.output_size = 1;           // Number of targets
config.hidden_layer_size = 160;   // Hidden state size
config.num_heads = 4;             // Attention heads
config.quantiles = {0.1f, 0.5f, 0.9f}; // Quantiles to predict

// Define input types
config.input_obs_loc = {0};       // Target column indices
config.known_regular_input_idx = {1, 2, 3, 4}; // Known future inputs
config.static_input_loc = {};     // Static features (if any)

// Create model
auto model = TemporalFusionTransformer(config);

// Prepare data (TFTData structure)
TFTData train_data = load_your_data();
TFTData valid_data = load_your_validation_data();

// Train model
TFTTrainer trainer(model, config);
trainer.train(train_data, valid_data);

// Save model
model->save("trained_model.pt");
```

### Making Predictions

```cpp
// Load trained model
auto model = TemporalFusionTransformer(config);
model->load("trained_model.pt");

// Prepare test data
auto test_inputs = prepare_test_data(); // [batch_size, seq_len, features]

// Make predictions
auto predictions = model->predict(test_inputs);

// Extract quantile predictions
auto all_preds = predictions.predictions; // [batch, forecast_steps, output*quantiles]
auto median_preds = all_preds.narrow(2, 1 * config.output_size, config.output_size);

// Access attention weights for interpretability
auto attention = predictions.attention_weights;
auto variable_importance = attention.static_flags;      // Static variable importance
auto temporal_attention = attention.decoder_self_attn;  // Self-attention weights
```

### Data Format

The TFT expects data in the following format:

```cpp
struct TFTData {
    torch::Tensor inputs;        // [batch_size, time_steps, num_features]
    torch::Tensor outputs;       // [batch_size, forecast_steps, output_size]  
    torch::Tensor active_entries; // [batch_size, forecast_steps, output_size] (1.0 for valid, 0.0 for padding)
    torch::Tensor time;          // [batch_size, forecast_steps, 1] (time indices)
    torch::Tensor identifiers;   // [batch_size, forecast_steps, 1] (entity IDs)
};
```

### Input Types

Configure different input types based on your data:

- **TARGET**: The variable you want to forecast (observed only historically)
- **OBSERVED_INPUT**: Historical inputs not known in the future  
- **KNOWN_INPUT**: Inputs known for both historical and future periods
- **STATIC_INPUT**: Time-invariant features (same for all timesteps)
- **CATEGORICAL**: Categorical variables (require embedding)

## Model Configuration

### Key Hyperparameters

```cpp
struct TFTConfig {
    // Architecture
    int hidden_layer_size = 160;    // Hidden state dimension
    int num_heads = 4;              // Multi-head attention heads
    int num_stacks = 1;             // Number of transformer stacks
    float dropout_rate = 0.1f;      // Dropout rate
    
    // Training  
    float learning_rate = 1e-3f;    // Adam learning rate
    float max_gradient_norm = 1.0f; // Gradient clipping
    int batch_size = 64;            // Training batch size
    int num_epochs = 100;           // Maximum training epochs
    
    // Forecasting
    std::vector<float> quantiles = {0.1f, 0.5f, 0.9f}; // Quantiles to predict
    int total_time_steps = 192;     // Input sequence length
    int num_encoder_steps = 168;    // Historical context length
};
```

## Interpretability Features

The TFT provides several interpretability features:

1. **Variable Selection Weights**: Importance of each input variable
2. **Temporal Attention**: Which historical time steps are most relevant
3. **Static vs Dynamic**: Separation of time-invariant and time-varying effects
4. **Quantile Predictions**: Uncertainty quantification

## Performance Tips

1. **GPU Usage**: Enable CUDA for faster training on large datasets
2. **Batch Size**: Adjust based on GPU memory (32-128 typical range)
3. **Sequence Length**: Longer sequences improve accuracy but increase memory usage
4. **Hidden Size**: Scale with dataset complexity (80-320 typical range)

## Differences from Original 

This implementation maintains full fidelity to the original TFT paper, including:
- ✅ LSTM + Self-Attention hybrid architecture  
- ✅ Variable Selection Networks
- ✅ Gated Residual Networks
- ✅ Interpretable Multi-Head Attention
- ✅ Quantile Loss Functions
- ✅ Static and Dynamic Variable Processing

## Citation

If you use this implementation, please cite the original TFT paper:

```bibtex
@article{lim2021temporal,
  title={Temporal fusion transformers for interpretable multi-horizon time series forecasting},
  author={Lim, Bryan and Arik, Sercan O and Loeff, Nicolas and Pfister, Tomas},
  journal={International Journal of Forecasting},
  volume={37},
  number={4},
  pages={1748--1764},
  year={2021},
  publisher={Elsevier}
}
```

## License

This implementation follows the original Apache 2.0 license.
