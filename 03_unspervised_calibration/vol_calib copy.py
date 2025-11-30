

"""vol_calib.py

Unsupervised latent-volatility model for joint Heston calibration using
TCN encoder + GRU decoder.

This file is written to integrate *conceptually* with your existing
`sv_models.py` (Heston/Bates models) and `cos_pricers.py` (COS option
pricers), but avoids hard-wiring exact signatures so you can adapt it to
your implementation easily.

High-level design
-----------------
- Encoder: TCN over returns (and optionally extra features) -> latent code z
- Decoder: GRU conditioned on z -> variance path v_t (t = 0..T-1)
- Loss: you plug this v_t into your Heston model and COS pricers to build
  a joint objective based on
    * returns likelihood under P,
    * Heston dynamics penalty (CIR-consistency),
    * option-pricing penalty via COS under Q.

You can run and extend this inside a Jupyter notebook by importing
`LatentVolatilityModel` and writing a custom training loop that calls
into your `HestonModel` and COS pricers.

The bottom of this file includes an example training skeleton.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility: simple TCN building blocks
# ---------------------------------------------------------------------------


class Chomp1d(nn.Module):
    """Chomp off extra padding at the end to keep causality and length.

    If we use padding = (kernel_size - 1) * dilation, Conv1d will produce
    extra time steps on the right; we "chomp" them off to recover original
    sequence length.
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size]


class TemporalBlock(nn.Module):
    """A basic TCN block with dilated causal convolutions and residual skip.

    Args:
        in_channels: input feature channels
        out_channels: output feature channels
        kernel_size: Conv1d kernel size (odd is typical)
        dilation: dilation factor
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)

        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (batch, C_in, T)
        returns: (batch, C_out, T)
        """

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TCNEncoder(nn.Module):
    """TCN encoder over returns (and optional extra features).

    Input format:
        x: (batch, T, input_dim)  # time-major last

    Output:
        z: (batch, latent_dim)

    The encoder applies a stack of TemporalBlocks and then global pooling
    (mean + max) over time, followed by a small MLP to produce the latent
    code z.
    """

    def __init__(
        self,
        input_dim: int,
        channels: Tuple[int, ...] = (32, 32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.0,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        layers = []
        c_in = input_dim
        for i, c_out in enumerate(channels):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            c_in = c_out
        self.tcn = nn.Sequential(*layers)

        # Map pooled representation to latent code z
        self.to_latent = nn.Sequential(
            nn.Linear(2 * c_in, max(2 * latent_dim, c_in)),
            nn.ReLU(),
            nn.Linear(max(2 * latent_dim, c_in), latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences.

        Args:
            x: (batch, T, input_dim)

        Returns:
            z: (batch, latent_dim)
        """
        # Conv1d expects (batch, C, T)
        x_cnn = x.transpose(1, 2)  # (batch, input_dim, T)
        h = self.tcn(x_cnn)  # (batch, C_last, T)

        # Global pooling over time
        h_mean = h.mean(dim=-1)
        h_max, _ = h.max(dim=-1)
        h_cat = torch.cat([h_mean, h_max], dim=-1)  # (batch, 2*C_last)

        z = self.to_latent(h_cat)
        return z


# ---------------------------------------------------------------------------
# GRU decoder: latent -> volatility path
# ---------------------------------------------------------------------------


class GRUDecoder(nn.Module):
    """GRU decoder that generates a variance path v_t from a latent z.

    We use the latent vector z to initialize the GRU hidden state, and then
    unroll a sequence of length T to produce log-variance or variance.

    Args:
        latent_dim: dimension of latent code z
        hidden_dim: GRU hidden size
        output_len: length of the variance path to generate (T)
        output_mode: "logvar" | "var".
            - "logvar": network predicts log v_t, we exponentiate.
            - "var": network predicts v_t with softplus to enforce positivity.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_len: int,
        output_mode: str = "logvar",
    ) -> None:
        super().__init__()
        assert output_mode in {"logvar", "var"}
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_len = output_len
        self.output_mode = output_mode

        self.latent_to_h0 = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.h_to_out = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate a variance path from latent codes.

        Args:
            z: (batch, latent_dim)

        Returns:
            v: (batch, T, 1)  # variance path, strictly positive
        """
        batch_size, latent_dim = z.shape
        assert latent_dim == self.latent_dim

        # Initialize GRU hidden state from z
        h0 = torch.tanh(self.latent_to_h0(z)).unsqueeze(0)  # (1, batch, hidden_dim)

        # Repeat z at each time step as input (global conditioning)
        z_seq = z.unsqueeze(1).repeat(1, self.output_len, 1)  # (batch, T, latent_dim)

        h_seq, _ = self.gru(z_seq, h0)  # (batch, T, hidden_dim)
        out = self.h_to_out(h_seq)  # (batch, T, 1)

        if self.output_mode == "logvar":
            v = torch.exp(out).clamp_min(1e-10)
        else:  # "var"
            v = F.softplus(out) + 1e-10
        return v


# ---------------------------------------------------------------------------
# Combined latent volatility model
# ---------------------------------------------------------------------------


class LatentVolatilityModel(nn.Module):
    """Full TCN encoder + GRU decoder model.

    This model *only* produces a variance path v_t from input sequences. It
    does not itself know anything about Heston, P vs Q, or COS pricing. Those
    are supplied externally in your loss function.

    Typical input features:
        - returns r_t
        - (optional) realized variance proxy
        - (optional) option-implied features

    Input x must have shape (batch, T, input_dim).
    Output v has shape (batch, T, 1).
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        tcn_channels: Tuple[int, ...] = (32, 32, 64, 64),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        latent_dim: int = 32,
        decoder_hidden_dim: int = 64,
        output_mode: str = "logvar",
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.encoder = TCNEncoder(
            input_dim=input_dim,
            channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            latent_dim=latent_dim,
        )
        self.decoder = GRUDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_len=seq_len,
            output_mode=output_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, T, input_dim)

        Returns:
            v: (batch, T, 1) variance path
        """
        assert x.shape[1] == self.seq_len, "Input sequence length mismatch"
        z = self.encoder(x)
        v = self.decoder(z)
        return v


# ---------------------------------------------------------------------------
# Dataclasses for option data and configuration
# ---------------------------------------------------------------------------


@dataclass
class OptionBatch:
    """Container for option data at a given time index.

    This is intentionally generic; you can adapt it to your actual data
    structures. In the simplest case, all options in the batch correspond to
    the *final* time in the sequence, with maturity tau and strike K.

    Shapes:
        S:    (batch,)        current spot at option observation time
        K:    (batch, n_opt)  strikes
        tau:  (batch, n_opt)  time-to-maturity in years
        price:(batch, n_opt)  observed option prices (call or put)
    """

    S: torch.Tensor
    K: torch.Tensor
    tau: torch.Tensor
    price: torch.Tensor
    is_call: bool = True


@dataclass
class HestonParams:
    """Minimal container for Heston P-measure parameters.

    These are used to derive Q-measure parameters via your equity and
    volatility risk premia. You likely already have a richer HestonModel
    class in sv_models.py; this dataclass is mainly to make the interfaces
    flexible and explicit.
    """

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    xi_s: float  # equity risk premium parameter
    xi_v: float  # volatility risk premium parameter


# ---------------------------------------------------------------------------
# Example joint loss hooks (to be adapted to your sv_models + cos_pricers)
# ---------------------------------------------------------------------------


class JointHestonLoss(nn.Module):
    """Joint loss wrapper for Heston calibration with latent volatility.

    This module demonstrates how you *could* structure the objective using
    your latent volatility model, a Heston model implementation, and COS
    option pricers. It is deliberately written in a generic way, with hooks
    that you can fill using your actual `sv_models.HestonModel` and
    `cos_pricers` functions.

    The actual numerical details of the return likelihood and CIR dynamics
    are left to you, because they depend strongly on how your HestonModel is
    implemented (Euler vs QE, log-returns vs levels, etc.).
    """

    def __init__(
        self,
        latent_model: LatentVolatilityModel,
        heston_params: HestonParams,
        lambda_dyn: float = 1.0,
        lambda_opt: float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_model = latent_model
        self.heston_params = heston_params
        self.lambda_dyn = lambda_dyn
        self.lambda_opt = lambda_opt

    # -------------------- Hooks you should adapt --------------------

    def returns_loglik(  # pragma: no cover - to be customized
        self,
        S: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute *negative* log-likelihood of returns under P.

        Args:
            S: (batch, T+1) spot path
            v: (batch, T, 1) variance path (output of latent model)
            dt: time step size

        Returns:
            neg_loglik: scalar tensor

        NOTE: This is a placeholder. In your implementation, you can:
            - compute log-returns r_t = log(S_{t+1}/S_t)
            - use your HestonModel Euler/QE transition density to model
              r_t | v_t and sum up -log p(r_t | v_t, params)
        """
        # Simple Gaussian proxy example (you should replace with your own):
        r = torch.log(S[:, 1:] / S[:, :-1]).unsqueeze(-1)  # (batch, T, 1)
        # Mean under P: (r - q + xi_s * v_t) dt  ~ 0 for proxy
        # For simplicity we use zero mean and variance = v_t * dt
        var = (v * dt).clamp_min(1e-8)
        neg_loglik = 0.5 * (r**2 / var + torch.log(2 * math.pi * var))
        return neg_loglik.sum() / S.shape[0]

    def cir_dynamics_penalty(  # pragma: no cover - to be customized
        self,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Penalty for deviations from CIR/Heston variance dynamics.

        Args:
            v: (batch, T, 1) variance path
            dt: time step size

        Returns:
            penalty: scalar tensor

        Example implementation (you can refine using exact CIR moments):
            v_{t+1} - v_t - kappa(theta - v_t) dt should be small.
        """
        kappa = self.heston_params.kappa
        theta = self.heston_params.theta

        v_t = v[:, :-1, :]  # (batch, T-1, 1)
        v_tp1 = v[:, 1:, :]
        drift = kappa * (theta - v_t) * dt
        residual = (v_tp1 - v_t - drift)
        penalty = (residual**2).mean()
        return penalty

    def option_pricing_penalty(
        self,
        v: torch.Tensor,
        opt: OptionBatch,
        r: float,
        q: float,
        cos_module: Any,
    ) -> torch.Tensor:
        """Option pricing penalty using true Heston risk-neutral (Q) parameters.

        Under Q:
            kappa_Q = kappa + xi_v
            theta_Q = (kappa * theta) / kappa_Q
        """
        # --------------------------------------------------------------
        # 1. Extract final-step variance and option fields
        # --------------------------------------------------------------
        v_T = v[:, -1, 0]  # (batch,)
        S = opt.S          # (batch,)
        K = opt.K          # (batch, n_opt)
        tau = opt.tau      # (batch, n_opt)
        price_mkt = opt.price  # (batch, n_opt)

        if not hasattr(cos_module, "price_call_COS"):
            raise RuntimeError("cos_module must expose price_call_COS(...)")

        # --------------------------------------------------------------
        # 2. Build true Q-measure Heston parameters
        # --------------------------------------------------------------
        hp = self.heston_params
        kappa_Q = hp.kappa + hp.xi_v
        theta_Q = (hp.kappa * hp.theta) / float(kappa_Q)

        # Build a qpar object matching what cos_pricers expects
        class Rates:
            def __init__(self, r, q):
                self.r = r
                self.q = q

        class QParams:
            def __init__(self, kappa_Q, theta_Q, sigma, rho, v0, r, q):
                self.kappa_Q = float(kappa_Q)
                self.theta_Q = float(theta_Q)
                self.sigma = float(sigma)
                self.rho = float(rho)
                self.v0 = float(v0)
                self.rates = Rates(float(r), float(q))

        qpar = QParams(
            kappa_Q=kappa_Q,
            theta_Q=theta_Q,
            sigma=hp.sigma,
            rho=hp.rho,
            v0=hp.v0,
            r=r,
            q=q,
        )

        # --------------------------------------------------------------
        # 3–4. Call COS pricer per batch element (no batch vectorization)
        # --------------------------------------------------------------
        batch_size = v_T.shape[0]
        price_list = []
        for i in range(batch_size):
            S_i = S[i].unsqueeze(0)           # (1,)
            v_i = v_T[i].unsqueeze(0)         # (1,)
            K_i = K[i]                        # (n_opt,)
            # tau_i: use scalar maturity for this row (all strikes share same τ)
            tau_i = tau[i, 0]

            price_i = cos_module.price_call_COS(
                model="heston",
                S=S_i,
                v=v_i,
                tau=tau_i,
                qpar=qpar,
                K=K_i,
            )  # (n_opt,)

            # Ensure 1D (n_opt,) before stacking
            if price_i.dim() > 1:
                price_i = price_i.reshape(-1)
            price_list.append(price_i)

        # Stack to (batch, n_opt)
        price_model = torch.stack(price_list, dim=0)

        # ---- Stability clamps (temporary) ----
        # Prevent NaNs/Infs from COS when model is untrained
        price_model = price_model.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)
        price_mkt = price_mkt.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)
        K = K.nan_to_num(nan=0.0)
        S = S.nan_to_num(nan=1.0)
        tau = tau.nan_to_num(nan=1e-4)

        # --------------------------------------------------------------
        # 5. Standardize device and market tensors
        # --------------------------------------------------------------
        device = v.device
        price_mkt = price_mkt.to(device)
        K = K.to(device)
        S = S.to(device)
        tau = tau.to(device)

        # Market Vega
        if not hasattr(opt, "vega"):
            raise RuntimeError("OptionBatch must contain market vega as opt.vega")
        # Clamp market vega to avoid exploding z early in training
        vega_mkt = opt.vega.to(device).clamp_min(1e-4).nan_to_num(nan=1e-4)

        # --------------------------------------------------------------
        # 6. Vega-normalized error
        # --------------------------------------------------------------
        z = (price_model - price_mkt) / vega_mkt
        z = z.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)
        z = z.clamp(min=-1e6, max=1e6)

        # Sample variance (MLE): sigma^2 = mean(z^2)
        sigma2 = z.pow(2).mean().clamp_min(1e-6)

        # Gaussian negative log-likelihood
        opt_nll = 0.5 * (z.pow(2) / sigma2 + torch.log(sigma2))
        return opt_nll.mean()

    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        S: torch.Tensor,
        dt: float,
        opt: Optional[OptionBatch] = None,
        r: float = 0.0,
        q: float = 0.0,
        cos_module: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute joint loss components and total loss.

        Args:
            x: (batch, T, input_dim) input features (e.g. returns)
            S: (batch, T+1) spot paths corresponding to x
            dt: time step size between observations
            opt: optional OptionBatch for option pricing penalty
            r, q: risk-free and dividend rate
            cos_module: your COS pricer module

        Returns:
            dict with keys:
                - "loss": total scalar loss
                - "neg_loglik": returns negative log-likelihood
                - "dyn_penalty": variance dynamics penalty
                - "opt_penalty": option pricing penalty (0 if opt is None)
        """
        v = self.latent_model(x)  # (batch, T, 1)

        neg_loglik = self.returns_loglik(S=S, v=v, dt=dt)
        dyn_penalty = self.cir_dynamics_penalty(v=v, dt=dt)

        if opt is not None and cos_module is not None:
            opt_penalty = self.option_pricing_penalty(
                v=v,
                opt=opt,
                r=r,
                q=q,
                cos_module=cos_module,
            )
        else:
            opt_penalty = torch.tensor(0.0, device=v.device)

        # --------------------------------------------------------------
        # Normalized joint loss (returns & options weighted by sample size)
        # --------------------------------------------------------------
        M = x.shape[1]                 # number of return observations (T)
        if opt is not None:
            N = opt.K.shape[1]         # number of options
        else:
            N = 0

        # Per-observation averages
        neg_loglik_norm = neg_loglik / M
        opt_penalty_norm = opt_penalty / N if N > 0 else 0.0
        dyn_penalty_norm = dyn_penalty   # already mean() over time

        # Automatic weights based on sample size
        total = M + N
        w_r = M / total
        w_o = N / total if N > 0 else 0.0

        # Final normalized loss
        loss = (
            w_r * neg_loglik_norm
            + w_o * opt_penalty_norm
            + self.lambda_dyn * dyn_penalty_norm
        )

        # Attach model_prices and qpar if option pricing was run
        model_prices = None
        qpar = None
        if opt is not None and cos_module is not None:
            # recompute price_model used inside option_pricing_penalty
            v_T = v[:, -1, 0]
            batch_size = v_T.shape[0]
            price_list = []
            # Build Q-measure Heston parameters as in option_pricing_penalty
            hp = self.heston_params
            kappa_Q = hp.kappa + hp.xi_v
            theta_Q = (hp.kappa * hp.theta) / float(kappa_Q)
            class Rates:
                def __init__(self, r, q):
                    self.r = r
                    self.q = q
            class QParams:
                def __init__(self, kappa_Q, theta_Q, sigma, rho, v0, r, q):
                    self.kappa_Q = float(kappa_Q)
                    self.theta_Q = float(theta_Q)
                    self.sigma = float(sigma)
                    self.rho = float(rho)
                    self.v0 = float(v0)
                    self.rates = Rates(float(r), float(q))
            qpar = QParams(
                kappa_Q=kappa_Q,
                theta_Q=theta_Q,
                sigma=hp.sigma,
                rho=hp.rho,
                v0=hp.v0,
                r=r,
                q=q,
            )
            S_batch = opt.S
            K_batch = opt.K
            tau_batch = opt.tau
            for i in range(batch_size):
                S_i = S_batch[i].unsqueeze(0)
                v_i = v_T[i].unsqueeze(0)
                K_i = K_batch[i]
                tau_i = tau_batch[i,0]
                price_i = cos_module.price_call_COS(
                    model="heston", S=S_i, v=v_i, tau=tau_i, qpar=qpar, K=K_i,
                )
                if price_i.dim() > 1:
                    price_i = price_i.reshape(-1)
                price_list.append(price_i)
            model_prices = torch.stack(price_list, dim=0)

        return {
            "loss": loss,
            "neg_loglik": neg_loglik.detach(),
            "dyn_penalty": dyn_penalty.detach(),
            "opt_penalty": opt_penalty.detach(),
            "model_prices": model_prices,
            "qpar": qpar if opt is not None else None,
        }


# ---------------------------------------------------------------------------
# Example: how to use this in a Jupyter notebook
# ---------------------------------------------------------------------------


def example_training_step(
    model: JointHestonLoss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any],
    dt: float,
    cos_module: Optional[Any] = None,
) -> Dict[str, float]:
    """One training step on a batch.

    Args:
        model: JointHestonLoss instance
        optimizer: torch optimizer
        batch: dict with keys
            - "x": (batch, T, input_dim)
            - "S": (batch, T+1)
            - optionally, "opt" (OptionBatch)
            - optionally, "r", "q" (floats)
        dt: time step size
        cos_module: COS pricer module

    Returns:
        dict of float metrics for logging.
    """
    x = batch["x"]
    S = batch["S"]
    opt = batch.get("opt", None)
    r = batch.get("r", 0.0)
    q = batch.get("q", 0.0)

    optimizer.zero_grad()
    out = model(x=x, S=S, dt=dt, opt=opt, r=r, q=q, cos_module=cos_module)
    loss = out["loss"]
    loss.backward()
    optimizer.step()

    metrics = {}
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            metrics[k] = float(v.item())
        else:
            metrics[k] = v  # keep tensors/vectors/dicts as-is
    return metrics


if __name__ == "__main__":
    # Minimal smoke test: create a dummy model and run a forward pass.
    batch_size = 4
    T = 64
    input_dim = 2  # e.g. [returns, realized variance]

    latent_model = LatentVolatilityModel(
        input_dim=input_dim,
        seq_len=T,
        tcn_channels=(16, 16, 32),
        tcn_kernel_size=3,
        tcn_dropout=0.1,
        latent_dim=16,
        decoder_hidden_dim=32,
        output_mode="logvar",
    )

    heston_params = HestonParams(
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
        xi_s=0.0,
        xi_v=0.0,
    )

    joint_model = JointHestonLoss(
        latent_model=latent_model,
        heston_params=heston_params,
        lambda_dyn=1.0,
        lambda_opt=1.0,
    )

    # Dummy batch
    x = torch.randn(batch_size, T, input_dim)
    S0 = torch.full((batch_size, 1), 100.0)
    noise = 0.01 * torch.randn(batch_size, T)
    S_path = S0 * torch.exp(noise.cumsum(dim=1))  # (batch, T)
    S = torch.cat([S0, S_path], dim=1)  # (batch, T+1)

    batch = {"x": x, "S": S}
    dt = 1.0 / 252.0

    optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-3)

    metrics = example_training_step(
        model=joint_model,
        optimizer=optimizer,
        batch=batch,
        dt=dt,
        cos_module=None,  # plug in your cos_pricers module when ready
    )

    print("Smoke-test metrics:", metrics)