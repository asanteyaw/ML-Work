# MLE-NGARCH: Maximum Likelihood Estimation Benchmark for NGARCH Parameter Estimation

## Overview

This project implements the traditional Maximum Likelihood Estimation (MLE) approach for estimating NGARCH (Nonlinear Generalized Autoregressive Conditional Heteroskedasticity) model parameters. It serves as the benchmark for comparison against the novel TFT-based approach in the companion project `../TransformerGARCH/`.

## Purpose

This MLE implementation provides:
- **Baseline performance metrics** for NGARCH parameter estimation
- **Traditional econometric approach** using likelihood maximization
- **Reference implementation** for validating neural network approaches
- **Computational efficiency benchmarks** for large-scale parameter estimation

## NGARCH Model Specification

### Mathematical Formulation

The NGARCH model implemented follows:

**Conditional Variance Equation:**
```
h_t = ω + φ(h_{t-1} - ω) + α·h_{t-1}·(z²_{t-1} - 2γz_{t-1} - 1)
```

**Return Equation:**
```
x_t = r - d + λ√h_t - 0.5h_t + √h_t·z_t
```

Where:
- `z_t ~ N(0,1)` are standardized residuals
- `r` = risk-free rate (0.019)
- `d` = dividend yield (0.012)

### Parameters

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| **Omega** | ω | Long-run variance level | [0.000325, 0.000524] |
| **Alpha** | α | ARCH coefficient | [0.048, 0.057] |
| **Phi** | φ | Persistence parameter | [0.8, 0.84] |
| **Lambda** | λ | Risk premium | [0.025, 0.043] |
| **Gamma** | γ | Asymmetry parameter | [0.1, 0.6] |

## Project Structure

```
MLE/
├── src/                          # Source code
│   ├── main.cpp                 # Main MLE estimation pipeline
│   └── utils.cpp                # Utility functions
├── include/                      # Header files
│   ├── Model.h                  # NGARCH model implementation
│   ├── Functionals.h            # Mathematical functionals
│   ├── utils.h                  # Utility declarations
│   └── scaler.h                 # Parameter scaling utilities
├── build/                        # Build directory
│   └── nrech                    # Executable (after building)
├── losses/                       # Training loss outputs
│   ├── ngarch_loss1.csv
│   ├── ngarch_loss2.csv
│   └── *.csv                    # Other loss files
├── models/                       # Model outputs
│   └── model_synthetic_ml.pt    # Trained parameters
├── synth_batch.slurm            # SLURM batch script
├── slurm-*.out                  # SLURM output logs
├── CMakeLists.txt               # Build configuration
├── README.md                    # This file
└── .gitignore                   # Git ignore rules
```

## Algorithm

### Parameter Initialization
```cpp
// Random parameter initialization within bounds
omega = generate_random_values(0.000325, 0.000524, n_batch);
alpha = generate_random_values(0.048, 0.057, n_batch);
phi = generate_random_values(0.8, 0.84, n_batch);
lambda = generate_random_values(0.025, 0.043, n_batch);
gamma = generate_random_values(0.1, 0.6, n_batch);
```

### Parameter Constraints
- **Slope transformation** applied to enforce parameter bounds
- **Inverse transformation** for parameter recovery
- **Automatic differentiation** via PyTorch for gradient computation

### Optimization Process
1. **Forward Pass**: Compute conditional variances and standardized residuals
2. **Likelihood Calculation**: Evaluate log-likelihood function
3. **Backward Pass**: Compute gradients w.r.t. parameters
4. **Parameter Update**: Adam optimizer step
5. **Constraint Enforcement**: Apply transformations to maintain bounds

## Requirements

- **LibTorch**: PyTorch C++ API (>= 1.8.0)
- **PyBind11**: Python-C++ binding library
- **CMake**: Build system (>= 3.12)
- **C++17**: Modern C++ standard
- **fmt**: Modern C++ formatting library
- **SLURM**: Optional for cluster execution

## Installation & Building

### Local Build

```bash
cd MLE
mkdir build && cd build
cmake ..
make -j4
```

### Cluster Build (with SLURM)

```bash
# Submit batch job
sbatch synth_batch.slurm
```

## Usage

### Basic Execution

```bash
# Run MLE estimation
./build/nrech
```

### Configuration

Edit `src/main.cpp` to modify:

```cpp
// Training parameters
int64_t n_epochs = 6000;           // Training epochs
size_t n_batch = 50000;           // Number of time series
size_t ts_len = 20000;            // Length of each series
double tr = 0.1;                  // Train/validation split

// Model configuration
std::string vol_type = "EGARCH";   // Volatility model type
const std::string c_path = "../checkpoints/ngarch/synthetic/ml/";
const std::string t_path = "../losses/ngarch_loss2.csv";
```

### SLURM Configuration

Edit `synth_batch.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=mle_ngarch
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out

./build/nrech
```

## Output Files

### Model Parameters
- **Location**: `models/model_synthetic_ml.pt`
- **Format**: PyTorch tensor file
- **Contents**: Optimized NGARCH parameters

### Training Losses
- **Location**: `losses/ngarch_loss2.csv`
- **Format**: CSV file with epoch, loss values
- **Usage**: Convergence analysis and diagnostics

### SLURM Logs
- **Pattern**: `slurm-[job_id].out`
- **Contents**: Training progress, convergence info, final parameters

## Performance Characteristics

### Computational Complexity
- **Time Complexity**: O(n_batch × ts_len × n_epochs)
- **Space Complexity**: O(n_batch × ts_len)
- **Typical Runtime**: 6-24 hours for 50K series × 20K timesteps

### Memory Requirements
- **Base Memory**: ~8GB for model and data
- **Per Batch**: ~1MB per 1000 time series
- **Recommended**: 64GB RAM for large-scale experiments

### Convergence Properties
- **Epochs to Convergence**: 3,000-6,000 epochs typically
- **Loss Function**: Negative log-likelihood
- **Optimizer**: Adam with learning rate 0.001
- **Convergence Criteria**: Loss plateau detection

## Benchmarking Against TFT

### Performance Metrics

| Metric | MLE Benchmark | TFT Target |
|--------|---------------|------------|
| **Parameter Recovery MSE** | Baseline | 30-50% improvement |
| **Training Time** | 6-24 hours | 2-6 hours |
| **Memory Usage** | 64GB | 32GB |
| **Convergence Stability** | Moderate | High |
| **Interpretability** | Model-based | Attention-based |

### Comparative Analysis

**MLE Strengths:**
- Theoretically grounded approach
- Well-established in econometrics
- Direct parameter interpretation
- No architecture tuning needed

**MLE Limitations:**
- Sensitive to initialization
- Long convergence times
- Limited scalability
- Assumption-dependent

## Troubleshooting

### Build Issues

**LibTorch Not Found:**
```bash
export CMAKE_PREFIX_PATH=/path/to/libtorch:$CMAKE_PREFIX_PATH
cmake ..
```

**PyBind11 Missing:**
```bash
pip install pybind11[global]
```

### Runtime Issues

**Memory Errors:**
- Reduce `n_batch` in `main.cpp`
- Increase system memory or use cluster

**Convergence Problems:**
- Adjust learning rate in optimizer
- Increase `n_epochs`
- Check parameter initialization ranges

**SLURM Issues:**
```bash
# Check job status
squeue -u $USER

# View job details
scontrol show job [job_id]

# Cancel job if needed
scancel [job_id]
```

## Data Generation

The benchmark includes synthetic NGARCH data generation:

```cpp
// Generate time series with known parameters
auto [returns, variances] = model->simulate_ngarch(true_params, ts_len);

// Parameters are drawn from specified ranges
// Returns follow NGARCH dynamics
// Variances computed from conditional variance equation
```

## Validation

### Parameter Recovery Test
1. Generate data with known parameters
2. Run MLE estimation
3. Compare estimated vs. true parameters
4. Compute recovery error metrics

### Likelihood Function Test
```cpp
// Verify likelihood computation
auto [z, h] = model->ngarch_eq(params, returns);
auto loglik = model->loglikelihood(z, h);
```

## Future Enhancements

- **Multi-asset NGARCH**: Extend to multivariate models
- **Alternative Optimizers**: BFGS, L-BFGS-B implementation
- **Parallel Processing**: OpenMP/MPI parallelization
- **Real Data Integration**: Market data preprocessing
- **Robust Estimation**: Outlier-resistant methods

## References

1. **NGARCH Models**: Engle & Ng (1993), "Measuring and testing the impact of news on volatility"
2. **Maximum Likelihood**: Hamilton (1994), "Time Series Analysis"
3. **PyTorch C++**: https://pytorch.org/cppdocs/
4. **Econometric Theory**: Greene (2018), "Econometric Analysis"

## License

MIT License - see LICENSE file for details.

## Contact

For questions about the MLE benchmark implementation, please open an issue in the repository or contact the research team.

---

**Note**: This MLE benchmark is designed to work in conjunction with the TFT-based approach in `../TransformerGARCH/`. Both projects share common data formats and evaluation metrics for fair comparison.
