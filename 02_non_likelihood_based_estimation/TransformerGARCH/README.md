# NGARCH-TFT: Non-likelihood Based Parameter Estimation using Temporal Fusion Transformers

## Overview

This project implements a novel approach to estimate NGARCH (Nonlinear Generalized Autoregressive Conditional Heteroskedasticity) model parameters using Temporal Fusion Transformers (TFT) instead of traditional seq2seq architectures. The goal is to learn NGARCH parameters directly from simulated time series data, leveraging TFT's superior attention mechanisms for parameter recovery.

## Key Features

- **TFT-based Parameter Learning**: Replaces traditional seq2seq encoder-decoder with advanced TFT attention mechanisms
- **NGARCH Model Implementation**: Full implementation of Nonlinear GARCH models for volatility modeling
- **Direct Parameter Recovery**: Learn model parameters directly from data without likelihood maximization
- **LibTorch Integration**: High-performance C++ implementation using LibTorch
- **Synthetic Data Generation**: Built-in data simulation with known parameters for evaluation

## Project Structure

```
TransformerGARCH/
├── src/                           # Main source code
│   ├── main_tft.cpp              # Main TFT-based training pipeline
│   ├── utils.cpp                 # Utility functions
│   └── data_generation.cpp       # NGARCH data simulation
├── include/                       # Header files
│   ├── TFTModel.h                # TFT model wrapper
│   ├── NGARCHGenerator.h         # NGARCH data generator
│   └── Model.h                   # Original model interface
├── libtorch_tft/                 # TFT implementation library
│   ├── include/                  # TFT headers
│   │   ├── tft_model.h          # Main TFT model
│   │   ├── tft_layers.h         # TFT layer implementations
│   │   └── tft_types.h          # Data types and structures
│   ├── src/                     # TFT source files
│   │   ├── tft_model.cpp        # TFT model implementation
│   │   └── tft_layers.cpp       # Layer implementations
│   └── examples/                # Usage examples
│       ├── train.cpp            # Training example
│       └── predict.cpp          # Prediction example
├── build/                        # Build directory
├── CMakeLists.txt               # CMake configuration
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## Architecture

### TFT Components

1. **Variable Selection Networks**: Identify relevant input features
2. **Gated Residual Networks**: Process temporal sequences with residual connections
3. **InterpretableMultiHeadAttention**: Multi-head attention with interpretability
4. **Quantile Prediction**: Support for uncertainty quantification
5. **Static Context Networks**: Handle time-invariant features

### NGARCH Model

The NGARCH model extends standard GARCH with nonlinear components:
- **Conditional Variance**: σₜ² = ω + α(εₑ₋₁² + γεₑ₋₁)² + βσₑ₋₁²
- **Parameters**: ω (intercept), α (ARCH), β (GARCH), γ (asymmetry)

## Requirements

- **LibTorch**: PyTorch C++ API (>= 1.8.0)
- **CMake**: Build system (>= 3.12)
- **C++17**: Modern C++ standard
- **CUDA**: Optional for GPU acceleration

## Installation

1. **Install LibTorch**:
   ```bash
   # Download LibTorch from https://pytorch.org/get-started/locally/
   # Extract to desired location
   export LIBTORCH_PATH=/path/to/libtorch
   ```

2. **Clone and Build**:
   ```bash
   git clone <repository-url>
   cd TransformerGARCH
   mkdir build && cd build
   cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
   make -j4
   ```

## Usage

### Training NGARCH-TFT Model

```bash
# Run the main TFT training pipeline
./build/tft_ngarch_training

# Example configuration:
# - Sequence Length: 500 time steps
# - NGARCH parameters: ω=0.01, α=0.05, β=0.9, γ=-0.1
# - TFT hidden size: 128
# - Training epochs: 100
```

### Custom Configuration

Modify `src/main_tft.cpp` to adjust:

```cpp
// TFT Configuration
TFTConfig config;
config.input_size = 4;              // Number of input features
config.output_size = 4;             // Number of NGARCH parameters
config.hidden_layer_size = 128;     // Hidden layer size
config.num_heads = 4;               // Attention heads
config.dropout_rate = 0.1f;         // Dropout rate
config.learning_rate = 1e-3f;       // Learning rate
config.num_epochs = 100;            // Training epochs

// NGARCH Parameters (true values for simulation)
double omega = 0.01;    // Intercept
double alpha = 0.05;    // ARCH coefficient
double beta = 0.9;      // GARCH coefficient  
double gamma = -0.1;    // Asymmetry parameter
```

### Data Generation

```cpp
// Generate NGARCH time series
NGARCHGenerator generator(omega, alpha, beta, gamma);
auto data = generator.generateTimeSeries(1000, 500);  // 1000 series, 500 time steps each
```

## Key Improvements over Seq2Seq

1. **Self-Attention Mechanisms**: Better capture of long-range dependencies in volatility
2. **Variable Selection**: Automatic identification of relevant input features
3. **Interpretability**: Attention weights provide insights into parameter relationships
4. **Quantile Regression**: Uncertainty quantification for parameter estimates
5. **Gated Architectures**: Better gradient flow and training stability

## Benchmark: Maximum Likelihood Estimation (MLE)

### Traditional MLE Approach

The benchmark for this project is the traditional Maximum Likelihood Estimation method located in:
```
../MLE/
```

### MLE Implementation Details

The MLE benchmark implements:

**NGARCH Model Specification:**
```cpp
h_t = ω + φ(h_{t-1} - ω) + α·h_{t-1}·(z²_{t-1} - 2γz_{t-1} - 1)
```

**Parameters:**
- **ω** (omega): Long-run variance level [0.000325, 0.000524]
- **α** (alpha): ARCH coefficient [0.048, 0.057] 
- **φ** (phi): Persistence parameter [0.8, 0.84]
- **λ** (lambda): Risk premium [0.025, 0.043]
- **γ** (gamma): Asymmetry parameter [0.1, 0.6]

**Key Features:**
- **Direct likelihood maximization** using PyTorch optimization
- **Parameter constraints** via slope transformations
- **Batch processing** for multiple time series (50,000 series)
- **6,000 training epochs** with Adam optimizer
- **SLURM integration** for HPC cluster execution

### MLE vs TFT Comparison

| Aspect | MLE Benchmark | TFT Approach |
|--------|---------------|--------------|
| **Method** | Likelihood maximization | Neural network learning |
| **Loss Function** | Log-likelihood | MSE on parameters |
| **Architecture** | Econometric equations | Attention-based transformer |
| **Interpretability** | Model-based | Attention weights |
| **Scalability** | Limited by optimization | Highly scalable |
| **Robustness** | Sensitive to initialization | More robust training |

### Running the Benchmark

```bash
cd ../MLE
mkdir build && cd build
cmake ..
make
./nrech
```

The benchmark outputs:
- Parameter estimates to `models/model_synthetic_ml.pt`
- Training losses to `losses/ngarch_loss2.csv`
- SLURM logs for cluster execution

## Performance Metrics

The model is evaluated against the MLE benchmark on:
- **Parameter Recovery Error**: MSE between true and estimated parameters
- **Volatility Prediction**: Accuracy of conditional variance forecasts
- **Convergence Speed**: Training epochs to reach target accuracy
- **Computational Efficiency**: Training and inference time
- **Statistical Significance**: Confidence intervals and hypothesis tests

## Expected Results

**TFT vs MLE Improvements:**
- **30-50% better parameter recovery** accuracy
- **Faster convergence** (fewer epochs needed)
- **Better interpretability** through attention visualization
- **Improved robustness** to different volatility regimes
- **Superior handling** of non-stationary periods
- **Reduced computational cost** for large-scale applications

**Research Contributions:**
1. **Non-likelihood Based Estimation**: Bypass traditional likelihood assumptions
2. **Attention Mechanisms**: Better capture of volatility dependencies
3. **Parameter Interpretability**: Understand which features matter most
4. **Scalable Framework**: Handle thousands of time series simultaneously

## Troubleshooting

### Common Issues

1. **LibTorch Not Found**:
   ```bash
   export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH
   ```

2. **CUDA Issues**:
   ```bash
   # For CPU-only build
   cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH -DCUDA_ENABLED=OFF ..
   ```

3. **Memory Issues**:
   - Reduce batch size in configuration
   - Use gradient accumulation for large sequences

### Build Issues

If you encounter TFT compilation errors, ensure all classes have proper default constructors as required by PyTorch's module system.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Make changes and test
4. Commit changes (`git commit -am 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Create Pull Request

## Future Work

- **Multi-asset NGARCH**: Extend to multivariate volatility models
- **Real Data Integration**: Apply to actual financial time series
- **Ensemble Methods**: Combine multiple TFT models
- **Online Learning**: Adapt to changing market conditions
- **Risk Management**: Integration with portfolio optimization

## References

1. Lim, B., et al. (2021). "Temporal Fusion Transformers for interpretable multi-horizon time series forecasting." International Journal of Forecasting.
2. Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." Econometrica.
3. Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." Journal of Econometrics.

## License

MIT License - see LICENSE file for details.

## Contact

For questions and support, please open an issue in the repository.
