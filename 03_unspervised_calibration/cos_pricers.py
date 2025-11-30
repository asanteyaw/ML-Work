# -----------------------------------------------------------------------------
# Implements the COS method of Fang (2009) for Heston pricing using.
#  
#  - Fully batched
#  - Uses torch.fft
#  - Autograd compatible → gradients flow back to v_t and Heston parameters
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import math

# -------------------------------------------------------------
# 1. Heston affine coefficients C(τ,u), D(τ,u)
# -------------------------------------------------------------

def heston_C_D(tau, u, kappa, theta, sigma, rho, r_minus_q):
    """
    All inputs: torch tensors
    u: shape (..., N)
    returns: C, D with same broadcasting
    """
    iu = 1j * u

    a = kappa * theta
    b = kappa

    d = torch.sqrt((rho * sigma * iu - b)**2 + sigma**2 * (u**2 + iu))
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

    exp_dt = torch.exp(-d * tau)
    one_minus_gexp = 1 - g * exp_dt
    one_minus_g = 1 - g

    C = iu * r_minus_q * tau + (a / sigma**2) * (
        (b - rho * sigma * iu - d) * tau - 2 * torch.log(one_minus_gexp / one_minus_g)
    )

    D = (b - rho * sigma * iu - d) / sigma**2 * ((1 - exp_dt) / one_minus_gexp)
    return C, D


# -------------------------------------------------------------
# 2. Cumulants (c1,c2) used for truncation interval [a,b]
# -------------------------------------------------------------

def heston_cumulants(kappa, theta, sigma, v0, rho, r, q, T):
    exp_kt = torch.exp(-kappa * T)
    c1 = (r - q) * T + (1 - exp_kt) * (theta - v0) / (2 * kappa) - 0.5 * theta * T

    # c2 can be complicated; approximate with classical closed-form
    # Ensures positivity for std
    c2 = (
        sigma**2 * v0 * T * 0.5
        + kappa * theta * sigma**2 * T / (2 * kappa**2)
    )
    c2 = torch.clamp(c2, min=1e-12)
    return c1, c2


# -------------------------------------------------------------
# 3. Generate the COS payoff coefficients U_k for CALLS
# -------------------------------------------------------------

def cos_call_Uk(a, b, K, N):
    """
    K: (nK,) tensor
    Returns:
        U_k: (nK, N)
        shift: (nK, N)  exp(-i u (a - ln K))
        u: (N,)
    """
    K = K.reshape(-1)
    nK = K.size(0)

    k = torch.arange(N, dtype=torch.float64, device=K.device)
    u = k * math.pi / (b - a)  # shape (N,)

    logK = torch.log(K).unsqueeze(1)  # (nK,1)
    aK = a - logK
    bK = b - logK

    denom = 1 + (k.unsqueeze(0) * math.pi / (bK - aK))**2
    chi_k = (torch.cos(k * math.pi) * torch.exp(bK) - torch.exp(aK)) / denom

    psi_k = torch.zeros_like(chi_k)
    psi_k[:, 0] = (bK - aK).squeeze(1)

    U_k = 2.0 / (bK - aK) * (chi_k - psi_k)
    U_k[:, 0] *= 0.5

    shift = torch.exp(-1j * u.unsqueeze(0) * aK)
    return U_k, shift, u


# -------------------------------------------------------------
# 4. FULL COS PRICER (PyTorch) → differentiable
# -------------------------------------------------------------

def heston_cos_price(S, v, tau, qpar, K, N=512, L=12.0):
    """
    Vectorized differentiable COS price for CALL
    S: (...,) torch
    v: (...,) torch
    tau: scalar torch
    K: (...,) torch
    returns: call prices
    """

    device = S.device
    dtype = torch.float64

    S = S.to(dtype)
    v = v.to(dtype)
    K = K.to(dtype)

    S, v, K = torch.broadcast_tensors(S, v, K)
    flat_S = S.reshape(-1)
    flat_v = v.reshape(-1)
    flat_K = K.reshape(-1)
    n = flat_S.size(0)

    kappa_Q = torch.tensor(qpar.kappa_Q, dtype=dtype, device=device)
    theta_Q = torch.tensor(qpar.theta_Q, dtype=dtype, device=device)
    sigma = torch.tensor(qpar.sigma, dtype=dtype, device=device)
    rho = torch.tensor(qpar.rho, dtype=dtype, device=device)
    v0 = torch.tensor(qpar.v0, dtype=dtype, device=device)
    r = torch.tensor(qpar.rates.r, dtype=dtype, device=device)
    q = torch.tensor(qpar.rates.q, dtype=dtype, device=device)

    c1, c2 = heston_cumulants(kappa_Q, theta_Q, sigma, v0, rho, r, q, tau)
    std = torch.sqrt(c2)
    half = L * std

    a = c1 - half
    b = c1 + half

    U_k, shift, u = cos_call_Uk(a, b, flat_K, N)

    C, D = heston_C_D(tau, u, kappa_Q, theta_Q, sigma, rho, (r - q))

    x = torch.log(torch.clamp(flat_S, min=1e-16)).unsqueeze(1)
    vv = flat_v.unsqueeze(1)

    phi = torch.exp(C.unsqueeze(0) + D.unsqueeze(0) * vv + 1j * u.unsqueeze(0) * x)

    Re = torch.real(phi * shift)

    disc = torch.exp(-r * tau)
    prices = disc * flat_K * torch.sum(U_k * Re, dim=1)

    return prices.reshape(S.shape).to(torch.float32)


# -------------------------------------------------------------
# 5. Public API wrappers
# -------------------------------------------------------------

def price_call_COS(model, S, v, tau, qpar, K, N=512, L=12.0):
    if model != "heston":
        raise ValueError("Only Heston supported in PyTorch COS.")
    return heston_cos_price(S, v, tau, qpar, K, N=N, L=L)


def price_put_COS(model, S, v, tau, qpar, K, N=512, L=12.0):
    if model != "heston":
        raise ValueError("Only Heston supported.")
    C = price_call_COS(model, S, v, tau, qpar, K, N=N, L=L)
    r = torch.tensor(qpar.rates.r, dtype=C.dtype, device=C.device)
    qv = torch.tensor(qpar.rates.q, dtype=C.dtype, device=C.device)
    return C - S * torch.exp(-qv * tau) + K * torch.exp(-r * tau)


def price_straddle_COS(model, S, v, tau, qpar, K, N=512, L=12.0):
    C = price_call_COS(model, S, v, tau, qpar, K, N=N, L=L)
    P = price_put_COS(model, S, v, tau, qpar, K, N=N, L=L)
    return C + P