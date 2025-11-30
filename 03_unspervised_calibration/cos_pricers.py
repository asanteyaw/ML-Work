# cos_pricers.py (minimal, fully vectorized, de-bloated)
# -----------------------------------------------------------------------------
# Vectorized COS pricing & Greeks for Heston and Bates under a chosen ELMM Q.
# Single API for both scalar and batch inputs — everything broadcasts.
# No imports from sv_models; qpar/bq are duck-typed containers.
# Exports:
#   - u_and_greeks_COS(model, S, v, tau, qpar, K, N=512, L=12.0)
#   - price_call_COS / price_put_COS / price_straddle_COS
#   - Compatibility wrappers:
#       heston_u_and_greeks_COS_paths, bates_u_and_greeks_COS_paths
#       heston_call_u_and_greeks_vec,  bates_call_u_and_greeks_vec
# -----------------------------------------------------------------------------

from typing import Tuple, Literal
import numpy as np

# =========================
#   Core affine building blocks
# =========================

def _heston_C_D(
    tau: float,
    u: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    rminusq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Heston affine coefficients C(τ,u), D(τ,u) for ln S."""
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    a = kappa * theta
    b = kappa
    d = np.sqrt((rho * sigma * iu - b) ** 2 + (sigma**2) * (u**2 + iu))
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

    exp_dt = np.exp(-d * tau)
    one_minus_gexp = 1 - g * exp_dt
    one_minus_g = 1 - g

    C = iu * rminusq * tau + (a / (sigma**2)) * (
        (b - rho * sigma * iu - d) * tau - 2.0 * np.log(one_minus_gexp / one_minus_g)
    )
    D = ((b - rho * sigma * iu - d) / (sigma**2)) * ((1 - exp_dt) / one_minus_gexp)
    return C, D


def _cumulants_heston(kappa, theta, sigma, v0, rho, r, q, T):
    """Cumulants c1,c2 used to set [a,b]."""
    exp_kt = np.exp(-kappa * T)
    c1 = (r - q) * T + (1 - exp_kt) * (theta - v0) / (2 * kappa) - 0.5 * theta * T
    c2 = (1 / (8 * kappa**3)) * (
        sigma * T * kappa * exp_kt * (v0 - theta) * (8 * kappa * rho - 4 * sigma)
        + kappa * sigma * rho * (1 - exp_kt) * (16 * theta - 8 * v0)
        + 2 * theta * kappa * T * (-4 * kappa * rho * sigma + sigma**2 + 4 * kappa**2)
        + sigma**2 * ((theta - 2 * v0) * exp_kt**2 + theta * (6 * exp_kt - 7) + 2 * v0)
        + 8 * kappa**2 * (v0 - theta) * (1 - exp_kt)
    )
    return float(c1), float(max(c2, 1e-12))


def _truncation_from_cumulants(c1: float, c2: float, L: float) -> Tuple[float, float]:
    std = np.sqrt(max(c2, 1e-12))
    half = L * std
    return c1 - half, c1 + half


# =========================
#   COS payoff coefficients for CALL (vectorized over strikes)
# =========================

def _Uk_shift_call(a: float, b: float, K: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (U_k, shift, u) with shapes:
       U_k   : (nK, N)
       shift : (nK, N)
       u     : (N,)
    Works for scalar or array K (nK,)."""
    K = np.asarray(K, dtype=float).reshape(-1)
    nK = K.size
    k = np.arange(N, dtype=float)
    u = k * np.pi / (b - a)

    logK = np.log(K)[:, None]  # (nK,1)
    aK = (a - logK)
    bK = (b - logK)

    denom = 1.0 + (k[None, :] * np.pi / (bK - aK)) ** 2
    chi_k = (np.cos(k[None, :] * np.pi) * np.exp(bK) - np.exp(aK)) / denom

    # psi_k requires a special-case for k=0; for integer k≥1, sin(kπ)=0
    psi_k = np.zeros_like(chi_k, dtype=float)          # shape (nK, N)
    # k == 0 corresponds to the first column
    psi_k[:, 0] = (bK - aK)[:, 0]
    # for k >= 1, psi_k remains zero

    U_k = 2.0 / (bK - aK) * (chi_k - psi_k)  # (nK,N)
    U_k[:, 0] *= 0.5

    shift = np.exp(-1j * u[None, :] * aK)     # (nK,N)
    return U_k, shift, u


# =========================
#   Vectorized COS: price & (∂S, ∂v) for CALL under Heston/Bates
# =========================

def u_and_greeks_COS(
    model: Literal["heston", "bates"],
    S: np.ndarray,
    v: np.ndarray,
    tau: float,
    qpar,
    K: np.ndarray,
    N: int = 512,
    L: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized COS evaluation of (price, ∂_S, ∂_v) for CALL options.

    Inputs
    ------
    model : "heston" | "bates"
    S     : (...,) spot(s) at current time t
    v     : (...,) variance state(s)
    tau   : time-to-maturity (scalar)
    qpar  : HestonQ-like for heston, BatesQ-like for bates
            - heston: fields kappa_Q, theta_Q, sigma, rho, v0, rates.{r,q}
            - bates : object with .heston (above) and .jump.{m1, psi(u)}
    K     : (...,) strike(s) — broadcastable to S/v
    N, L  : COS parameters

    Returns
    -------
    price : (...,) call price(s)
    dS    : (...,) share delta(s)
    dv    : (...,) partial wrt instantaneous variance (via affine D-term)
    """
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    K = np.asarray(K, dtype=float)

    # Broadcast S, v, K to common 1D shape
    S_b, v_b, K_b = np.broadcast_arrays(S, v, K)
    out_shape = S_b.shape
    S_vec = S_b.reshape(-1)
    v_vec = v_b.reshape(-1)
    K_vec = K_b.reshape(-1)
    n = S_vec.size

    # Extract Heston-like parameters for truncation and continuous part
    if model == "heston":
        hq = qpar
        kappa = float(hq.kappa_Q); theta = float(hq.theta_Q)
        sigma = float(hq.sigma);    rho   = float(hq.rho)
        r = float(hq.rates.r);      q = float(hq.rates.q)
        vbar0 = float(getattr(hq, "v0", 0.0))
        m1 = 0.0
        jump_psi = None
    else:  # bates
        hq = qpar.heston
        kappa = float(hq.kappa_Q); theta = float(hq.theta_Q)
        sigma = float(hq.sigma);    rho   = float(hq.rho)
        r = float(hq.rates.r);      q = float(hq.rates.q)
        vbar0 = float(getattr(hq, "v0", 0.0))
        m1 = float(getattr(qpar.jump, "m1", 0.0))
        jump_psi = qpar.jump.psi  # function of u

    # Truncation interval from cumulants (path-independent)
    c1, c2 = _cumulants_heston(kappa, theta, sigma, vbar0, rho, r, q, tau)
    a, b = _truncation_from_cumulants(c1, c2, L)

    # COS payoff coeffs for all strikes (vectorized over K)
    U_k, shift, u = _Uk_shift_call(a, b, K_vec, N)  # (n,N), (n,N), (N,)

    # Affine continuous part (correct r-q by m1 for Bates)
    C, D = _heston_C_D(tau, u, kappa, theta, sigma, rho, (r - q) - m1)

    # Build CF φ for each path: exp(C + D v + i u ln S) * jump-term(if Bates)
    x = np.log(np.maximum(S_vec, 1e-300))[:, None]  # (n,1)
    vv = v_vec[:, None]                              # (n,1)

    base = np.exp(C[None, :] + D[None, :] * vv + 1j * u[None, :] * x)  # (n,N)
    if model == "bates":
        base = base * np.exp(tau * jump_psi(u))[None, :]

    # Common real part with strike shift
    Re = np.real(base * shift)                       # (n,N)

    # Prices
    disc = np.exp(-r * tau)
    prices = disc * K_vec * np.sum(U_k * Re, axis=1)  # (n,)

    # Delta: multiply integrand by (i u)/S
    factor_s = (1j * u[None, :]) / np.maximum(S_vec[:, None], 1e-12)
    dS = disc * K_vec * np.sum(U_k * np.real(base * shift * factor_s), axis=1)

    # dv: multiply integrand by D(τ,u)
    dv = disc * K_vec * np.sum(U_k * np.real(base * shift * D[None, :]), axis=1)

    # Reshape back
    return prices.reshape(out_shape), dS.reshape(out_shape), dv.reshape(out_shape)


# =========================
#   Convenience price-only (call/put/straddle) built on the unified greeks
# =========================

def price_call_COS(model: Literal["heston", "bates"], S, v, tau, qpar, K, N=512, L=12.0):
    p, _, _ = u_and_greeks_COS(model, S, v, tau, qpar, K, N=N, L=L)
    return p

def price_put_COS(model: Literal["heston", "bates"], S, v, tau, qpar, K, N=512, L=12.0):
    """Put via parity: P = C - S e^{-qτ} + K e^{-rτ} (vectorized)."""
    if model == "heston":
        r = float(qpar.rates.r); qv = float(qpar.rates.q)
    else:
        r = float(qpar.heston.rates.r); qv = float(qpar.heston.rates.q)
    C = price_call_COS(model, S, v, tau, qpar, K, N=N, L=L)
    return C - np.asarray(S, float) * np.exp(-qv * tau) + np.asarray(K, float) * np.exp(-r * tau)

def price_straddle_COS(model: Literal["heston", "bates"], S, v, tau, qpar, K, N=512, L=12.0):
    C = price_call_COS(model, S, v, tau, qpar, K, N=N, L=L)
    if model == "heston":
        r = float(qpar.rates.r); qv = float(qpar.rates.q)
    else:
        r = float(qpar.heston.rates.r); qv = float(qpar.heston.rates.q)
    P = C - np.asarray(S, float) * np.exp(-qv * tau) + np.asarray(K, float) * np.exp(-r * tau)
    return C + P


# =========================
#   Compatibility wrappers (so old code keeps running)
# =========================

def heston_u_and_greeks_COS_paths(S_vec, v_vec, tau, qpar, K, N=512, L=12.0):
    """Compat wrapper: returns (price, dS, dv) across paths for Heston."""
    return u_and_greeks_COS("heston", S_vec, v_vec, tau, qpar, K, N=N, L=L)

def bates_u_and_greeks_COS_paths(S_vec, v_vec, tau, bq, K, N=512, L=12.0):
    """Compat wrapper: returns (price, dS, dv) across paths for Bates."""
    return u_and_greeks_COS("bates", S_vec, v_vec, tau, bq, K, N=N, L=L)

def heston_call_u_and_greeks_vec(S_vec, v_vec, K, tau, qpar, N=512, L=12.0):
    """Alias expected by older benchmark scripts."""
    return u_and_greeks_COS("heston", S_vec, v_vec, tau, qpar, K, N=N, L=L)

def bates_call_u_and_greeks_vec(S_vec, v_vec, K, tau, bq, N=512, L=12.0):
    """Alias expected by older benchmark scripts."""
    return u_and_greeks_COS("bates", S_vec, v_vec, tau, bq, K, N=N, L=L)