# Unsupervised Heston Calibration via TCN–GRU Latent Volatility Model

This module implements a **fully unsupervised joint calibration procedure** for the Heston stochastic volatility model using:

- **TCN encoder** to extract information from historical returns and optional auxiliary features.
- **GRU decoder** to generate a latent variance path.
- **Joint objective** combining:
  - Return-based likelihood under the **physical measure** \( P \)
  - Variance dynamics (CIR) structural penalty
  - Option-price reconstruction error under the **risk-neutral measure** \( Q \) using COS pricing
- Integration hooks for user-provided:
  - `sv_models.py` (Heston/Bates simulators and measure conversions)
  - `cos_pricers.py` (COS Fourier-based option pricers)

The goal is to infer the *entire latent volatility path* without any Kalman filter, particle filter, or realized volatility labels.

---

## Project Structure

```
04_unspervised_calibration/
├── vol_calib.py          # Main TCN–GRU latent volatility model + joint loss
├── README.md             # This file
├── (expected) sv_models.py
└── (expected) cos_pricers.py
```

---

## Core Components

### 1. **LatentVolatilityModel**
A neural model that converts an input sequence (e.g., returns) into a variance path:
- **Encoder:** Temporal Convolutional Network (TCN)
- **Decoder:** GRU unrolled for \( T \) time steps

Output is a strictly positive variance sequence \( v_{0:T-1} \).

### 2. **JointHestonLoss**
Wraps the latent model and defines the full unsupervised objective:

- **Return log-likelihood (under P)**
- **CIR dynamics penalty**
- **Option reconstruction via COS (under Q)**

This loss is intentionally *modular*—you plug in your own Heston transition density and COS pricing code.

### 3. **example_training_step**
A ready-made function for one optimization step using PyTorch.

---

## How to Use in a Jupyter Notebook

### 1. Import the components
```python
from vol_calib import (
    LatentVolatilityModel,
    HestonParams,
    JointHestonLoss,
    OptionBatch,
    example_training_step,
)
import cos_pricers
```

### 2. Build the latent model
```python
latent = LatentVolatilityModel(
    input_dim=1,
    seq_len=T,
    latent_dim=32,
    decoder_hidden_dim=64,
)
```

### 3. Provide starting Heston parameters
```python
params = HestonParams(
    kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04,
    xi_s=0.0, xi_v=0.0
)
```

### 4. Combine into the joint calibration loss
```python
joint = JointHestonLoss(latent, params)
optimizer = torch.optim.Adam(joint.parameters(), lr=1e-3)
```

### 5. Prepare a batch and train
```python
metrics = example_training_step(
    model=joint,
    optimizer=optimizer,
    batch=batch_dict,
    dt=1/252,
    cos_module=cos_pricers,
)
print(metrics)
```

---

## Customization Points

You must adapt three functions in `JointHestonLoss` to your code:

1. **returns_loglik** — should call your Heston under-P transition density.
2. **cir_dynamics_penalty** — can be replaced with exact CIR conditional moment penalties (see your `sv_models.py`).
3. **option_pricing_penalty** — must build `qpar` using your volatility risk premium and call your COS pricer.

---

## Purpose of Approach

This module demonstrates how deep sequence models can act as **implicit latent-variable models** for stochastic processes.

---

## Contact
For issues, debugging, or integration questions, drop me a note.
# Unsupervised Heston Calibration via TCN–GRU Latent Volatility Model

This module provides a **complete framework** for performing *unsupervised joint calibration* of the Heston stochastic volatility model using:

- **TCN (Temporal Convolutional Network) encoder**
- **GRU decoder**
- **Unsupervised latent-volatility inference**
- **Return likelihood under the physical measure (P)**
- **Option price reconstruction under the risk-neutral measure (Q)** using **COS Fourier pricing**

The model learns a **latent volatility path** directly from market data (returns + options), with **no filters**, **no realized volatility labels**, and **no supervised learning**.

This README explains **exactly how to install dependencies**, **run experiments**, **structure your data**, and **extend the code**.

---

# Installation & Environment Setup

## 1. Navigate into the project folder
```bash
cd 04_unspervised_calibration
```

## 2. Create a Python virtual environment
You can use `venv` or `conda`. Here is the recommended `venv` setup:

```bash
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate.bat # Windows
```

## 3. Install all dependencies
```bash
pip install -r requirements.txt
```
This installs:
- PyTorch
- NumPy, SciPy, Pandas, yfinance
- Jupyter + IPython kernel
- Matplotlib + tqdm

## 4. Register the kernel for Jupyter notebooks
```bash
python -m ipykernel install --user --name heston-env
```

You will now be able to select **heston-env** inside Jupyter.

---

# Project Structure

```
04_unsupervised_calibration/
├── vol_calib.py          # Main model: TCN encoder + GRU decoder + joint loss
├── README.md             # This file
├── requirements.txt      # Dependencies
├── (expected) sv_models.py     # Your Heston/Bates QE/Euler simulators
└── (expected) cos_pricers.py   # Your COS Fourier option pricing code
```

---

# Core Components (in `vol_calib.py`)

## **1. LatentVolatilityModel**
A neural model mapping input sequences to variance paths:

- **Encoder:** TCN (dilated causal convolutions)
- **Decoder:** GRU generating a sequence of variances

```python
from vol_calib import LatentVolatilityModel
latent_model = LatentVolatilityModel(
    input_dim=1,      # e.g. returns only
    seq_len=T,        # number of time steps
    latent_dim=32,    # bottleneck
    decoder_hidden_dim=64,
)
```

Output:
```
v: (batch, T, 1)   # positive variance path
```

---

## **2. JointHestonLoss**
This wraps the latent volatility model and computes a **joint objective**:

- Return likelihood under **P**
- Variance dynamics penalty (CIR)
- Option pricing penalty using **COS** under **Q**

```python
from vol_calib import JointHestonLoss, HestonParams

params = HestonParams(
    kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04,
    xi_s=0.0, xi_v=0.0,
)

joint = JointHestonLoss(latent_model, params)
```

You will **customize three functions** inside `JointHestonLoss`:
1. `returns_loglik` → use your Heston under-P conditional density
2. `cir_dynamics_penalty` → enforce CIR structure using your `moments_CIR`
3. `option_pricing_penalty` → call your COS pricing function with Q-params

---

## **3. example_training_step**
A helper function for training:

```python
metrics = example_training_step(
    model=joint,
    optimizer=optimizer,
    batch=batch_dict,
    dt=1/252,
    cos_module=cos_pricers,
)
```

Returns:
```
{
  "loss": ...,
  "neg_loglik": ...,
  "dyn_penalty": ...,
  "opt_penalty": ...,
}
```

---

# Running the Model in Jupyter Notebook

Open a notebook:
```bash
jupyter notebook
```
Select the **heston-env** kernel.

## Example notebook workflow

### **1. Import everything**
```python
from vol_calib import (
    LatentVolatilityModel,
    HestonParams,
    JointHestonLoss,
    OptionBatch,
    example_training_step,
)
import cos_pricers
import torch
```

### **2. Load your spot paths & compute returns**
```python
S_paths = ...   # (batch, T+1)
returns = torch.log(S_paths[:,1:] / S_paths[:,:-1])
x = returns.unsqueeze(-1)  # (batch, T, 1)
```

### **3. Option data (if available)**
```python
opt_batch = OptionBatch(
    S=S_paths[:,-1],
    K=K_tensor,        # (batch, n_opt)
    tau=tau_tensor,    # (batch, n_opt)
    price=price_tensor # market prices
)
```

### **4. Build and run the model**
```python
latent = LatentVolatilityModel(input_dim=1, seq_len=T)
params = HestonParams(...)
joint = JointHestonLoss(latent, params)
optimizer = torch.optim.Adam(joint.parameters(), lr=1e-3)

batch = {"x": x, "S": S_paths, "opt": opt_batch, "r": 0.01, "q": 0.00}
metrics = example_training_step(joint, optimizer, batch, dt=1/252, cos_module=cos_pricers)
metrics
```

---

# Data Format Requirements

### Inputs to the model:
- `x`: `(batch, T, input_dim)`
  - typically `[returns]` or `[returns, realized_var]`
- `S`: `(batch, T+1)` spot path
- `opt` (optional): `OptionBatch`

### OptionBatch fields:
```
S    # spot at pricing time
K    # strikes
tau  # maturities
price  # observed market prices
```

Everything must be PyTorch tensors.

---

# Customization Guide (VERY IMPORTANT)

You must modify three functions in `JointHestonLoss` to match your code:

## returns_loglik
Replace the Gaussian proxy with your **Heston QE/Euler** transition:
- Use `simulate_under_P`
- Include equity risk premium parameter `xi_s`
- Use conditional density of log returns given `v_t`

## cir_dynamics_penalty
Replace finite-difference residual with CIR moment-based penalty:
- Use `moments_CIR`
- Penalize deviations from conditional mean and variance of the CIR process

## 3️option_pricing_penalty
Must:
- Convert P → Q parameters using `xi_v`
- Construct `qpar` object used by `cos_pricers`
- Call `price_call_COS(...)` or relevant function

These 3 functions allow **tight integration with your existing Heston/Bates code**.

---

# Advanced Usage (Recommended)

- Add additional features: IV slices, VIX proxies, realized kernels
- Add attention inside the encoder (TCN → Attention → Latent → GRU)
- Add priors on the latent vector for Bayesian calibration
- Use multiple option dates inside the batch
- Train multiple instruments jointly

---

# Debugging Tips

- Start with **returns-only** training (no options) to stabilize the decoder
- Verify variance positivity after decoding
- Plot generated `v_t` paths to ensure smoothness
- Check gradient flow using `torch.autograd.gradcheck`
- Use small learning rate (1e-3 → 1e-4)

---

# License
---</file>