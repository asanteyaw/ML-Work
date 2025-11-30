"""
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

You can copy/paste these blocks directly into a .ipynb file, or import them and use.
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
            loss = metrics.get("loss")
            dyn = metrics.get("dyn_penalty")
            opt = metrics.get("opt_penalty")
            print(f"Epoch {epoch}: loss={loss}, dyn_penalty={dyn}, opt_penalty={opt}")

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

def plot_implied_vol_smile(joint_model, opt_batch, x=None, cos_module=cos_pricers, r=0.019, q=0.012):
    """
    Clean, professional volatility smile plot.
    Uses true Heston Q‑measure params + COS pricing.
    No artificial multipliers, no fabricated smile shape.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # 1. Obtain Q‑measure parameters from the model
    # -------------------------------------------------------------
    with torch.no_grad():
        S0_float = float(opt_batch.S[0])
        dummy_S = torch.tensor([[S0_float, S0_float]], dtype=torch.float32)
        out = joint_model(x=None, S=dummy_S, dt=1/252,
                          opt=opt_batch, r=r, q=q,
                          cos_module=cos_module)
        qpar = out["qpar"]

    # -------------------------------------------------------------
    # 2. Extract option data
    # -------------------------------------------------------------
    S0 = float(opt_batch.S[0].item())
    K = opt_batch.K[0].cpu().numpy()
    tau = float(opt_batch.tau[0,0].item())

    # Correct initial variance (P‑measure v0)
    v0 = torch.tensor([float(joint_model.heston_params.v0)], dtype=torch.float32)
    S_tensor = torch.tensor([S0], dtype=torch.float32)

    # -------------------------------------------------------------
    # 3. COS pricing for all strikes
    # -------------------------------------------------------------
    with torch.no_grad():
        model_prices_t = cos_module.heston_cos_price(
            S=S_tensor,
            v=v0,
            tau=tau,
            qpar=qpar,
            K=opt_batch.K[0],
        )
    model_prices = model_prices_t.cpu().numpy()

    # -------------------------------------------------------------
    # 4. Compute implied volatilities
    # -------------------------------------------------------------
    iv_model = []
    for i, strike in enumerate(K):
        price = float(model_prices[i])
        iv_val = implied_vol_call(S0, float(strike), r, tau, q, price)
        iv_model.append(iv_val)

    # -------------------------------------------------------------
    # 5. Plot the model smile
    # -------------------------------------------------------------
    plt.figure(figsize=(10,5))
    sns.lineplot(x=K, y=iv_model, linewidth=2, label="Model IV")
    plt.title("Model-Implied Vol Smile")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
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
    """
    Compute RMSE and IV-RMSE for option pricing errors.
    Fixed to properly extract model prices from COS pricing.
    """
    try:
        S = opt_batch.S
        # Ensure S is at least shape (1,2) 
        if S.dim() == 1:
            S = S.unsqueeze(0)
        if S.shape[1] == 1:
            S = torch.cat([S, S], dim=1)

        with torch.no_grad():
            # Get Q-parameters for COS pricing
            out = joint_model(x=None, S=S, dt=1/252,
                              opt=opt_batch, r=r, q=q,
                              cos_module=cos_module)
            qpar = out["qpar"]
            
            # Use COS pricer directly to get model prices
            S0 = float(opt_batch.S[0].item())
            tau = float(opt_batch.tau[0, 0].item())
            v0 = torch.tensor([joint_model.heston_params.v0], dtype=torch.float32)
            S_tensor = torch.tensor([S0], dtype=torch.float32)
            
            model_prices_t = cos_module.heston_cos_price(
                S=S_tensor,
                v=v0,
                tau=tau,
                qpar=qpar,
                K=opt_batch.K[0],
            )
            model_prices = model_prices_t.cpu().numpy()
        
        # Ensure arrays are 1D
        if model_prices.ndim > 1:
            model_prices = model_prices.flatten()
            
        market_prices = opt_batch.price[0].cpu().numpy()
        if market_prices.ndim > 1:
            market_prices = market_prices.flatten()
            
        vega_values = opt_batch.vega[0].cpu().numpy()
        if vega_values.ndim > 1:
            vega_values = vega_values.flatten()
        
        # Filter out NaN/Inf values
        valid_mask = ~(np.isnan(model_prices) | np.isnan(market_prices) | 
                      np.isnan(vega_values) | np.isinf(model_prices) | 
                      np.isinf(market_prices) | (vega_values == 0))
        
        if not np.any(valid_mask):
            print("Warning: No valid price comparisons found")
            return {"rmse": np.nan, "ivrmse": np.nan, "n_valid": 0}
        
        model_prices_clean = model_prices[valid_mask]
        market_prices_clean = market_prices[valid_mask]
        vega_clean = vega_values[valid_mask]
        
        # Compute errors
        price_errors = model_prices_clean - market_prices_clean
        rmse = np.sqrt(np.mean(price_errors**2))
        
        # IV-RMSE (vega-normalized errors)
        iv_errors = price_errors / vega_clean
        ivrmse = np.sqrt(np.mean(iv_errors**2))
        
        return {
            "rmse": float(rmse), 
            "ivrmse": float(ivrmse),
            "n_valid": int(np.sum(valid_mask)),
            "mean_model_price": float(np.mean(model_prices_clean)),
            "mean_market_price": float(np.mean(market_prices_clean))
        }
        
    except Exception as e:
        print(f"Error in compute_option_errors: {e}")
        return {"rmse": np.nan, "ivrmse": np.nan, "error": str(e)}

# =======================
# 20. VOLATILITY FORECASTING (1-STEP AHEAD)
# =======================

def forecast_volatility(latent_model, x):
    """
    Generate volatility forecast using the learned latent model.
    Fixed to handle tensor operations properly and provide multiple forecast methods.
    """
    try:
        with torch.no_grad():
            # Ensure input is properly shaped
            if x is None:
                raise ValueError("Input sequence x cannot be None for forecasting")
            
            # Get the learned variance path
            v = latent_model(x)  # Shape: (batch, T, 1)
            
            # Convert to numpy for analysis
            v_np = v[0, :, 0].cpu().numpy()
            
            # Multiple forecasting methods
            forecasts = {}
            
            # Method 1: Last value (naive)
            forecasts['last_value'] = float(v_np[-1])
            
            # Method 2: Simple moving average (last 5 values)
            window = min(5, len(v_np))
            forecasts['moving_avg_5'] = float(np.mean(v_np[-window:]))
            
            # Method 3: Exponential moving average
            alpha = 0.3
            ema = v_np[0]
            for i in range(1, len(v_np)):
                ema = alpha * v_np[i] + (1 - alpha) * ema
            forecasts['exp_moving_avg'] = float(ema)
            
            # Method 4: Linear trend extrapolation (last 10 points)
            if len(v_np) >= 10:
                x_trend = np.arange(len(v_np[-10:]))
                y_trend = v_np[-10:]
                coeffs = np.polyfit(x_trend, y_trend, 1)
                next_forecast = coeffs[0] * len(x_trend) + coeffs[1]
                forecasts['linear_trend'] = float(max(next_forecast, 1e-6))  # Ensure positive
            else:
                forecasts['linear_trend'] = forecasts['last_value']
            
            # Return dictionary with multiple forecasts
            forecasts['volatility_path'] = v_np.tolist()
            forecasts['path_length'] = len(v_np)
            forecasts['min_vol'] = float(np.min(v_np))
            forecasts['max_vol'] = float(np.max(v_np))
            forecasts['mean_vol'] = float(np.mean(v_np))
            forecasts['std_vol'] = float(np.std(v_np))
            
            return forecasts
            
    except Exception as e:
        print(f"Error in forecast_volatility: {e}")
        return {
            'last_value': np.nan,
            'moving_avg_5': np.nan,
            'exp_moving_avg': np.nan,
            'linear_trend': np.nan,
            'error': str(e)
        }

# =======================
# 21. RISK PREMIA ESTIMATION (ERP, VRP)
# =======================

def estimate_risk_premia(heston_params, realized_returns, learned_vol):
    """
    Estimate Equity Risk Premium (ERP) and Volatility Risk Premium (VRP).
    Fixed to handle various input types and provide comprehensive diagnostics.
    
    Parameters:
    - heston_params: HestonParams object or similar with theta/theta_Q attributes
    - realized_returns: array-like of realized stock returns
    - learned_vol: array-like of learned volatility values
    
    Returns:
    - Dictionary with ERP, VRP, and diagnostic information
    """
    try:
        # Convert inputs to numpy arrays for robustness
        if hasattr(realized_returns, 'cpu'):  # Handle torch tensors
            realized_returns = realized_returns.cpu().numpy()
        if hasattr(learned_vol, 'cpu'):  # Handle torch tensors
            learned_vol = learned_vol.cpu().numpy()
            
        realized_returns = np.asarray(realized_returns).flatten()
        learned_vol = np.asarray(learned_vol).flatten()
        
        # Remove NaN/Inf values
        valid_returns = realized_returns[~(np.isnan(realized_returns) | np.isinf(realized_returns))]
        valid_vol = learned_vol[~(np.isnan(learned_vol) | np.isinf(learned_vol))]
        
        if len(valid_returns) == 0:
            print("Warning: No valid realized returns found")
            erp = np.nan
        else:
            # ERP ~ mean(realized returns) - often annualized
            erp = float(np.mean(valid_returns))
            
        if len(valid_vol) == 0:
            print("Warning: No valid learned volatility values found")
            v_p = np.nan
        else:
            # P-measure expected volatility
            v_p = float(np.mean(valid_vol))

        # Q-measure long-run variance: support multiple parameter formats
        try:
            if hasattr(heston_params, "theta_Q"):
                v_q = float(heston_params.theta_Q)
            elif hasattr(heston_params, "theta"):
                v_q = float(heston_params.theta)
            else:
                # Try to extract from dict-like object
                if hasattr(heston_params, '__getitem__'):
                    if 'theta_Q' in heston_params:
                        v_q = float(heston_params['theta_Q'])
                    elif 'theta' in heston_params:
                        v_q = float(heston_params['theta'])
                    else:
                        raise KeyError("No theta_Q or theta found in heston_params")
                else:
                    raise AttributeError("heston_params has neither 'theta_Q' nor 'theta'")
        except Exception as e:
            print(f"Warning: Could not extract Q-measure theta: {e}")
            v_q = np.nan

        # Compute VRP (Volatility Risk Premium)
        if not (np.isnan(v_q) or np.isnan(v_p)):
            vrp = v_q - v_p
        else:
            vrp = np.nan
            
        # Additional diagnostics
        result = {
            "ERP": erp,
            "VRP": vrp,
            "P_measure_vol_mean": v_p,
            "Q_measure_vol_longrun": v_q,
            "realized_returns_mean": erp,
            "realized_returns_std": float(np.std(valid_returns)) if len(valid_returns) > 0 else np.nan,
            "learned_vol_mean": v_p,
            "learned_vol_std": float(np.std(valid_vol)) if len(valid_vol) > 0 else np.nan,
            "n_valid_returns": len(valid_returns),
            "n_valid_vol": len(valid_vol),
            "annualized_ERP": erp * 252 if not np.isnan(erp) else np.nan,  # Assuming daily data
            "annualized_return_vol": float(np.std(valid_returns)) * np.sqrt(252) if len(valid_returns) > 0 else np.nan
        }
        
        return result
        
    except Exception as e:
        print(f"Error in estimate_risk_premia: {e}")
        return {
            "ERP": np.nan, 
            "VRP": np.nan,
            "error": str(e)
        }

# =======================
# 22. ADDITIONAL HELPER FUNCTIONS FOR DOWNSTREAM TASKS
# =======================

def plot_implied_vol_smile_with_data(joint_model, opt_batch, x, cos_module=cos_pricers, r=0.019, q=0.012):
    """
    Explicit version of plot_implied_vol_smile that takes the returns data directly.
    This ensures you're using the exact yfinance returns data that trained your model.
    
    Usage after training:
    pnl.plot_implied_vol_smile_with_data(joint_model, opt_batch, x)
    """
    return plot_implied_vol_smile(joint_model, opt_batch, x=x, cos_module=cos_module, r=r, q=q)

def get_model_diagnostics(joint_model, opt_batch, x, cos_module=cos_pricers, r=0.019, q=0.012):
    """
    Comprehensive diagnostic function that combines all three problematic functions
    into one convenient call for downstream analysis.
    
    Returns a complete diagnostic report including:
    - Option pricing errors
    - Volatility forecasts  
    - Risk premia estimates
    - Model parameters and learned variance statistics
    """
    try:
        # Extract returns from the tensor for risk premia calculation
        realized_returns = x[0, :, 0].cpu().numpy()
        
        # Get learned variance path
        with torch.no_grad():
            learned_variance = joint_model.latent_model(x)
            learned_vol = learned_variance[0, :, 0].cpu().numpy()
        
        # Compute all diagnostics
        option_errors = compute_option_errors(joint_model, opt_batch, cos_module, r, q)
        vol_forecasts = forecast_volatility(joint_model.latent_model, x) 
        risk_premia = estimate_risk_premia(joint_model.heston_params, realized_returns, learned_vol)
        
        # Combine into comprehensive report
        diagnostics = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_summary": {
                "returns_length": len(realized_returns),
                "returns_mean": float(np.mean(realized_returns)),
                "returns_std": float(np.std(realized_returns)),
                "learned_vol_mean": float(np.mean(learned_vol)),
                "learned_vol_std": float(np.std(learned_vol)),
                "final_learned_vol": float(learned_vol[-1])
            },
            "option_pricing": option_errors,
            "volatility_forecasts": vol_forecasts,
            "risk_premia": risk_premia,
            "model_parameters": {
                "kappa": joint_model.heston_params.kappa,
                "theta": joint_model.heston_params.theta,
                "sigma": joint_model.heston_params.sigma,
                "rho": joint_model.heston_params.rho,
                "v0": joint_model.heston_params.v0,
                "xi_s": joint_model.heston_params.xi_s,
                "xi_v": joint_model.heston_params.xi_v
            }
        }
        
        return diagnostics
        
    except Exception as e:
        print(f"Error in get_model_diagnostics: {e}")
        return {"error": str(e)}

def print_training_summary(joint_model, opt_batch, x, history=None):
    """
    Print a nice summary of your training results and model performance.
    
    Usage:
    pnl.print_training_summary(joint_model, opt_batch, x, history)
    """
    print("="*60)
    print("UNSUPERVISED HESTON CALIBRATION - TRAINING SUMMARY")
    print("="*60)
    
    if history:
        final_loss = history[-1]['loss'] if len(history) > 0 else "N/A"
        print(f"Training epochs: {len(history)}")
        print(f"Final loss: {final_loss:.6f}" if isinstance(final_loss, (int, float)) else f"Final loss: {final_loss}")
        
        if len(history) > 1:
            improvement = (history[0]['loss'] - final_loss) / history[0]['loss'] * 100
            print(f"Loss improvement: {improvement:.2f}%")
    
    # Get comprehensive diagnostics
    diagnostics = get_model_diagnostics(joint_model, opt_batch, x)
    
    if "error" not in diagnostics:
        print(f"\nDATA SUMMARY:")
        print(f"- Returns series length: {diagnostics['data_summary']['returns_length']}")
        print(f"- Mean daily return: {diagnostics['data_summary']['returns_mean']:.6f}")
        print(f"- Return volatility: {diagnostics['data_summary']['returns_std']:.6f}")
        print(f"- Mean learned volatility: {diagnostics['data_summary']['learned_vol_mean']:.6f}")
        print(f"- Final learned volatility: {diagnostics['data_summary']['final_learned_vol']:.6f}")
        
        print(f"\nOPTION PRICING PERFORMANCE:")
        if 'rmse' in diagnostics['option_pricing']:
            print(f"- RMSE: {diagnostics['option_pricing']['rmse']:.4f}")
            print(f"- IV-RMSE: {diagnostics['option_pricing']['ivrmse']:.4f}")
            print(f"- Valid comparisons: {diagnostics['option_pricing']['n_valid']}")
        
        print(f"\nVOLATILITY FORECASTS:")
        if 'last_value' in diagnostics['volatility_forecasts']:
            print(f"- Last value: {diagnostics['volatility_forecasts']['last_value']:.6f}")
            print(f"- 5-day MA: {diagnostics['volatility_forecasts']['moving_avg_5']:.6f}")
            print(f"- Trend forecast: {diagnostics['volatility_forecasts']['linear_trend']:.6f}")
        
        print(f"\nRISK PREMIA:")
        if 'ERP' in diagnostics['risk_premia']:
            erp = diagnostics['risk_premia']['ERP']
            vrp = diagnostics['risk_premia']['VRP']
            print(f"- Equity Risk Premium: {erp:.6f}" if not np.isnan(erp) else "- Equity Risk Premium: N/A")
            print(f"- Volatility Risk Premium: {vrp:.6f}" if not np.isnan(vrp) else "- Volatility Risk Premium: N/A")
        
        print(f"\nMODEL PARAMETERS:")
        params = diagnostics['model_parameters']
        print(f"- κ (mean reversion): {params['kappa']:.4f}")
        print(f"- θ (long-run variance): {params['theta']:.6f}")
        print(f"- σ (vol-of-vol): {params['sigma']:.4f}")
        print(f"- ρ (correlation): {params['rho']:.4f}")
        print(f"- v₀ (initial variance): {params['v0']:.6f}")
    else:
        print(f"\nError generating diagnostics: {diagnostics['error']}")
    
    print("="*60)

# =======================
# END OF FILE EXTENSIONS
# =======================
