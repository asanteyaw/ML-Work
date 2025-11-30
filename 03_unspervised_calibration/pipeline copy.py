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

import torch, math
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", context="talk")
from statsmodels.tsa.stattools import acf

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
# 4. OPTION TABLE 
# =======================

def build_option_table():
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
# 5. CALCULATE BLACK-SCHOLES VEGA
# =======================
def black_scholes_vega(S, K, r, T, sigma, q):
  normpdf = lambda x: np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
  d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T));
  return np.exp(-q * T) * S * np.sqrt(T) * normpdf(d1); 

# =======================
# 5b. BLACK-SCHOLES CALL PRICE & IMPLIED VOL
# =======================

def black_scholes_call_price(S, K, r, T, sigma, q):
    """Black–Scholes European call price (vectorized over K, sigma).

    S, K, r, T, sigma, q can be numpy arrays or scalars with broadcasting.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Avoid division by zero
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    T = np.maximum(T, eps)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Standard normal CDF via error function
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

    return np.exp(-q * T) * S * cdf(d1) - np.exp(-r * T) * K * cdf(d2)


def implied_vol_call(S, K, r, T, q, price_mkt, max_iter=100, tol=1e-8):
    """Solve for Black–Scholes implied vol for a call option via Newton.

    All arguments are scalars. Returns a scalar implied vol.
    """
    # Initial guess
    sigma = 0.2
    for _ in range(max_iter):
        price = black_scholes_call_price(S, K, r, T, sigma, q)
        vega = black_scholes_vega(S, K, r, T, sigma, q)
        diff = price - price_mkt
        # If vega is tiny, break to avoid blow-up
        if abs(vega) < 1e-8:
            break
        sigma_new = sigma - diff / vega
        # Keep sigma in a reasonable range
        sigma_new = max(1e-4, min(5.0, sigma_new))
        if abs(sigma_new - sigma) < tol:
            sigma = sigma_new
            break
        sigma = sigma_new
    return sigma

# =======================
# 5. BUILD OPTIONBATCH
# =======================

def build_option_batch(option_table, S0, maturity_col="Jun2002"):
    # Maturity for June 2002 options
    today = datetime(2002, 4, 18)
    expiry = datetime(2002, 6, 21)
    tau_years = (expiry - today).days / 365

    # Extract strikes and market prices as numpy
    K_np = option_table["Strike"].values.astype(float)
    price_np = option_table[maturity_col].values.astype(float)

    # Rates used consistently with rest of pipeline
    r = 0.019
    q = 0.012
    T = tau_years

    # Compute market implied vols per strike via Black–Scholes
    sigma_iv = np.array([
        implied_vol_call(S0, float(K_np[i]), r, T, q, float(price_np[i]))
        for i in range(len(K_np))
    ], dtype=float)

    # Compute market vega per strike
    vega_np = black_scholes_vega(S0, K_np, r, T, sigma_iv, q)

    # Convert to torch tensors
    K = torch.tensor(K_np, dtype=torch.float32).unsqueeze(0)              # (1, n_opt)
    price = torch.tensor(price_np, dtype=torch.float32).unsqueeze(0)      # (1, n_opt)
    tau = torch.full_like(price, tau_years)                               # (1, n_opt)
    vega = torch.tensor(vega_np, dtype=torch.float32).unsqueeze(0)        # (1, n_opt)

    opt_batch = OptionBatch(
        S=torch.tensor([S0], dtype=torch.float32),
        K=K,
        tau=tau,
        price=price,
        is_call=True,
    )
    # Attach market vega as an attribute expected by vol_calib.JointHestonLoss
    opt_batch.vega = vega

    return opt_batch


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


# =======================
# 13. ADVANCED PLOTTING (SEABORN + GRIDS)
# =======================


def plot_loss(history):
    losses = [h["loss"] for h in history]
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(len(losses)), y=losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def plot_variance_path(latent_model, x):
    with torch.no_grad():
        v = latent_model(x)[0, :, 0].cpu().numpy()
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(len(v)), y=v)
    plt.title("Learned Latent Variance Path")
    plt.xlabel("t")
    plt.ylabel("v(t)")
    plt.grid(True)
    plt.show()

# =======================
# 14. REALIZED VOLATILITY VS LEARNED VOL
# =======================

def plot_realized_vs_learned(x, latent_model, window=21):
    returns = x[0,:,0].cpu().numpy()
    realized = []
    for i in range(len(returns)):
        if i < window:
            realized.append(np.nan)
        else:
            realized.append(np.sqrt(np.mean(returns[i-window:i]**2)))
    with torch.no_grad():
        v = latent_model(x)[0,:,0].cpu().numpy()
    plt.figure(figsize=(11,5))
    sns.lineplot(x=range(len(v)), y=v, label="Learned vol")
    sns.lineplot(x=range(len(realized)), y=realized, label="Realized vol")
    plt.title("Learned vs Realized Volatility")
    plt.xlabel("t")
    plt.ylabel("Vol")
    plt.grid(True)
    plt.legend()
    plt.show()

# =======================
# 15. IMPLIED VOL SMILE RECONSTRUCTION
# =======================

def plot_implied_vol_smile(joint_model, opt_batch, cos_module=cos_pricers, r=0.019, q=0.012):
    S = opt_batch.S
    K = opt_batch.K[0].cpu().numpy()
    tau = opt_batch.tau[0,0].item()
    with torch.no_grad():
        out = joint_model(x=None, S=S, dt=1/252, opt=opt_batch, r=r, q=q, cos_module=cos_module)
        model_prices = out["model_prices"].cpu().numpy() if "model_prices" in out else None
    if model_prices is None:
        return
    iv_model = []
    for i in range(len(K)):
        iv = implied_vol_call(float(S.item()), float(K[i]), r, tau, q, float(model_prices[0,i]))
        iv_model.append(iv)
    plt.figure(figsize=(10,5))
    sns.scatterplot(x=K, y=opt_batch.vega[0].cpu().numpy()*0 + 0, label="Market options", s=50)
    sns.lineplot(x=K, y=iv_model, label="Model IV")
    plt.title("Model-Implied Vol Smile")
    plt.xlabel("Strike")
    plt.ylabel("Implied Vol")
    plt.grid(True)
    plt.show()

# =======================
# 16. SAVE / LOAD MODEL
# =======================

def save_model(model, path="trained_joint_model.pt"):
    torch.save(model.state_dict(), path)


def load_model(model_class, path="trained_joint_model.pt"):
    model = model_class
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

# =======================
# 17. ACF OF LEARNED VOLATILITY
# =======================

def plot_vol_acf(latent_model, x, lags=30):
    with torch.no_grad():
        v = latent_model(x)[0,:,0].cpu().numpy()
    acf_vals = acf(v, nlags=lags, fft=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(range(lags+1)), y=acf_vals)
    plt.title("ACF of Learned Volatility")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True)
    plt.show()

# =======================
# 18. PCA REGIME DETECTION
# =======================
from sklearn.decomposition import PCA

def plot_vol_regime(latent_model, x):
    with torch.no_grad():
        v = latent_model(x)[0,:,0].cpu().numpy()
    pca = PCA(n_components=1)
    regime = pca.fit_transform(v.reshape(-1,1)).flatten()
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(len(regime)), y=regime)
    plt.title("Latent Volatility Regime Index (PCA)")
    plt.xlabel("t")
    plt.ylabel("Regime Index")
    plt.grid(True)
    plt.show()

# =======================
# 19. OUT-OF-SAMPLE OPTION RMSE & IVRMSE
# =======================

def compute_option_errors(joint_model, opt_batch, cos_module=cos_pricers,
                          r=0.019, q=0.012):
    with torch.no_grad():
        out = joint_model(x=None, S=opt_batch.S, dt=1/252,
                          opt=opt_batch, r=r, q=q,
                          cos_module=cos_module)
        model_prices = out["model_prices"][0].cpu().numpy()
    market_prices = opt_batch.price[0].cpu().numpy()
    rmse = np.sqrt(np.mean((model_prices - market_prices)**2))
    ivrmse = np.sqrt(np.mean(((model_prices - market_prices)/(opt_batch.vega[0].cpu().numpy()))**2))
    return {"rmse": rmse, "ivrmse": ivrmse}

# =======================
# 20. VOLATILITY FORECASTING (1-STEP AHEAD)
# =======================

def forecast_volatility(latent_model, x):
    with torch.no_grad():
        v = latent_model(x)[0,:,0].cpu().numpy()
    # Naive forecast = last value
    forecast = v[-1]
    return forecast

# =======================
# 21. RISK PREMIA ESTIMATION (ERP, VRP)
# =======================

def estimate_risk_premia(heston_params, realized_returns, learned_vol):
    # ERP ~ mean(realized returns)
    erp = float(np.mean(realized_returns))

    # VRP ~ E_Q[v] - E_P[v]
    # P-measure vol = learned_vol
    v_p = float(np.mean(learned_vol))

    # Q-measure vol approximation from Heston theta
    v_q = float(heston_params.theta)

    vrp = v_q - v_p
    return {"ERP": erp, "VRP": vrp}

# =======================
# END OF FILE EXTENSIONS
# =======================