"""
for_notebook.py
----------------
This script contains a FULL end‑to‑end unsupervised Heston calibration
pipeline designed specifically for copying into a Jupyter Notebook.

It is organized into numbered segments corresponding exactly to notebook
cells:
    1. Imports
    2. Download S&P 500 data
    3. Compute returns and slice window
    4. Build option table from textbook data
    5. Construct OptionBatch
    6. Build latent-volatility model
    7. Heston parameters
    8. Optimizer
    9. Single training step
    10. Multi-epoch loop
    11. Plot training loss
    12. Plot variance path

You can copy/paste these blocks directly into a .ipynb file.
"""

# =======================
# 1. IMPORTS
# =======================

import torch
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from vol_calib import (
    LatentVolatilityModel,
    HestonParams,
    JointHestonLoss,
    OptionBatch,
    example_training_step,
)

import cos_pricers  # Assumes your COS pricer file is available
from datetime import datetime


# =======================
# 2. DOWNLOAD S&P 500 DATA
# =======================

def load_sp500(start="2000-01-01", end="2002-04-18"):
    spx = yf.download("^GSPC", start=start, end=end, auto_adjust=True)
    spx["returns"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx = spx.dropna()
    return spx


# =======================
# 3. COMPUTE RETURNS & SLICE WINDOW
# =======================

def prepare_return_window(spx, T=252):
    x_returns = torch.tensor(spx["returns"].values[-T:], dtype=torch.float32)
    x = x_returns.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

    prices = torch.tensor(spx["Close"].values[-(T+1):], dtype=torch.float32)
    S_tensor = prices.unsqueeze(0)  # (1, T+1)

    S0 = float(prices[-1])
    return x, S_tensor, S0


# =======================
# 4. OPTION TABLE (FROM BOOK)
# =======================

def load_textbook_option_table():
    # TEMPLATE — USER FILLS VALUES
    option_table = pd.DataFrame({
        "Strike": [975, 995, 1025, 1050, 1075, 1090, 1100, 1110, 1120, 1125,
                    1130, 1135, 1140, 1150, 1160, 1170, 1175, 1200, 1225, 1250, 1275],

        # Fill these columns EXACTLY with data from your textbook table
        "May2002":  [161.60, 144.80, 120.10, 84.50, 64.30, 43.10, 35.60, 39.50,
                     22.90, 20.20, 28.00, 25.60, 13.30, 19.10, 15.30, 12.10,
                     10.90, None, None, None, None],

        "Jun2002":  [173.30, 157.00, 133.10, 100.70, 82.50, None, 65.50, None,
                     33.50, 30.70, None, 45.50, 23.20, 38.10, None, None, 27.70,
                     19.60, 13.20, None, 13.20],
    })

    option_table = option_table.dropna(how="all", axis=1)
    option_table = option_table.dropna(how="all", axis=0)
    return option_table


# =======================
# 5. BUILD OPTIONBATCH
# =======================

def build_option_batch(option_table, S0, maturity_col="Jun2002"):
    # Maturity for June 2002 options
    today = datetime(2002, 4, 18)
    expiry = datetime(2002, 6, 21)
    tau_years = (expiry - today).days / 365

    K = torch.tensor(option_table["Strike"].values, dtype=torch.float32).unsqueeze(0)
    price = torch.tensor(option_table[maturity_col].values,
                         dtype=torch.float32).unsqueeze(0)
    tau = torch.full_like(price, tau_years)

    return OptionBatch(
        S=torch.tensor([S0]),
        K=K,
        tau=tau,
        price=price,
        is_call=True,
    )


# =======================
# 6. BUILD LATENT MODEL
# =======================

def build_latent_model(T):
    return LatentVolatilityModel(
        input_dim=1,
        seq_len=T,
        tcn_channels=(32, 32, 64),
        latent_dim=32,
        decoder_hidden_dim=64,
        output_mode="logvar",
    )


# =======================
# 7. HESTON PARAMETERS
# =======================

def default_heston_params():
    return HestonParams(
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
        xi_s=0.0,
        xi_v=0.0,
    )


# =======================
# 8. OPTIMIZER
# =======================

def build_optimizer(model, lr=1e-3):
    return torch.optim.Adam(model.parameters(), lr=lr)


# =======================
# 9. RUN SINGLE TRAINING STEP
# =======================

def run_single_step(joint_model, optimizer, x, S_tensor, opt_batch):
    dt = 1/252
    batch = {"x": x, "S": S_tensor, "opt": opt_batch,
             "r": 0.019, "q": 0.012}
    return example_training_step(
        model=joint_model,
        optimizer=optimizer,
        batch=batch,
        dt=dt,
        cos_module=cos_pricers,
    )


# =======================
# 10. MULTI-EPOCH TRAINING LOOP
# =======================

def train_epochs(joint_model, optimizer, x, S_tensor, opt_batch,
                 epochs=300):
    history = []
    dt = 1/252

    for epoch in range(epochs):
        batch = {"x": x, "S": S_tensor, "opt": opt_batch,
                 "r": 0.019, "q": 0.012}
        metrics = example_training_step(
            model=joint_model,
            optimizer=optimizer,
            batch=batch,
            dt=dt,
            cos_module=cos_pricers,
        )
        history.append(metrics)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: {metrics}")

    return history


# =======================
# 11. PLOT TRAINING LOSS
# =======================

def plot_loss(history):
    losses = [h["loss"] for h in history]
    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()


# =======================
# 12. PLOT VARIANCE PATH
# =======================

def plot_variance_path(latent_model, x):
    with torch.no_grad():
        v = latent_model(x)[0, :, 0].numpy()
    plt.plot(v)
    plt.title("Learned Latent Variance Path")
    plt.show()


# END OF FILE