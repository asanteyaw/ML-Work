from __future__ import annotations
"""
Volatility-Integrated LSTM and GRU Models for Heston-Nandi and Component Heston-Nandi
"""

"""
Volatility-Integrated GRU Model for Heston-Nandi

Minimal TensorFlow 2 implementation:
- Single-layer GRU with hidden size 1
- Heston–Nandi variance enters as a dynamic bias in the GRU update gate
- Joint MLE over HN parameters and GRU weights using the HN-style likelihood

This is designed as a compact starting point; extensions (CHN, LSTM, multi-layer, options) can be added later.
"""


from typing import Optional

import tensorflow as tf


class VolIntegratedGRU(tf.keras.Model):
    """Single-asset, single-layer GRU with scalar hidden state.

    The Heston–Nandi variance h_t enters the GRU update gate as a dynamic bias.
    The hidden state η_t is interpreted as the conditional variance used in the
    return likelihood:

        R_t | η_t ~ N(λ * η_t, η_t)

    The Heston–Nandi recursion updates h_t using the same shock that drives
    returns, but with η_t in the inversion as in the proposal's NN-HN variant.
    """

    def __init__(self, input_dim: int, dtype: tf.dtypes.DType = tf.float32):
        super().__init__(dtype=dtype)

        # === Heston–Nandi parameters (trainable scalars) ===
        # Re-parameterised form: h_{t+1} = ω + ϕ(h_t - ω) + α(z_t^2 - 1 - 2 γ sqrt(h_t) z_t)
        self.omega = self.add_weight(
            name="omega", shape=(), initializer=tf.constant_initializer(0.000115867854986136), dtype=dtype
        )
        self.alpha = self.add_weight(
            name="alpha", shape=(), initializer=tf.constant_initializer(4.71105679172615e-06), dtype=dtype
        )
        self.phi = self.add_weight(
            name="phi", shape=(), initializer=tf.constant_initializer(0.962822073703399), dtype=dtype
        )
        self.lam = self.add_weight(
            name="lam", shape=(), initializer=tf.constant_initializer(2.43378692806841), dtype=dtype
        )
        self.gam = self.add_weight(
            name="gam", shape=(), initializer=tf.constant_initializer(186.082260369241), dtype=dtype
        )

        # === GRU parameters (hidden size = 1) ===
        self.input_dim = int(input_dim)

        # reset gate r_t
        self.W_r = self.add_weight(
            name="W_r", shape=(self.input_dim, 1), initializer="glorot_uniform", dtype=dtype
        )
        self.U_r = self.add_weight(
            name="U_r", shape=(1, 1), initializer="glorot_uniform", dtype=dtype
        )
        self.b_r = self.add_weight(
            name="b_r", shape=(1,), initializer="zeros", dtype=dtype
        )

        # update gate u_t with dynamic bias b_t = gamma_u * h_t
        self.W_u = self.add_weight(
            name="W_u", shape=(self.input_dim, 1), initializer="glorot_uniform", dtype=dtype
        )
        self.U_u = self.add_weight(
            name="U_u", shape=(1, 1), initializer="glorot_uniform", dtype=dtype
        )
        self.b_u = self.add_weight(
            name="b_u", shape=(1,), initializer="zeros", dtype=dtype
        )
        self.gamma_u = self.add_weight(  # scalar scale for dynamic bias
            name="gamma_u", shape=(), initializer="zeros", dtype=dtype
        )

        # candidate state n_t
        self.W_n = self.add_weight(
            name="W_n", shape=(self.input_dim, 1), initializer="glorot_uniform", dtype=dtype
        )
        self.U_n = self.add_weight(
            name="U_n", shape=(1, 1), initializer="glorot_uniform", dtype=dtype
        )
        self.b_n = self.add_weight(
            name="b_n", shape=(1,), initializer="zeros", dtype=dtype
        )

    @property
    def eps(self) -> tf.Tensor:
        return tf.constant(1e-8, dtype=self.dtype)

    def gru_step(self, x_t: tf.Tensor, h_vol_t: tf.Tensor, h_prev: tf.Tensor) -> tf.Tensor:
        """One GRU step with volatility-integrated update gate.

        Shapes (single asset):
            x_t:      [1, input_dim]
            h_vol_t:  [1, 1]   (Heston–Nandi variance h_t)
            h_prev:   [1, 1]   (previous hidden state)
        Returns:
            eta_t:    [1, 1]   (interpreted as variance, forced positive)
        """

        # reset gate r_t
        r_t = tf.sigmoid(
            tf.matmul(x_t, self.W_r) + self.b_r + tf.matmul(h_prev, self.U_r)
        )

        # update gate u_t with dynamic bias b_t = gamma_u * h_vol_t
        b_t = self.gamma_u * h_vol_t
        u_t = tf.sigmoid(
            tf.matmul(x_t, self.W_u) + self.b_u + tf.matmul(h_prev, self.U_u) + b_t
        )

        # candidate state n_t
        n_t = tf.tanh(
            tf.matmul(x_t, self.W_n) + self.b_n + r_t * (tf.matmul(h_prev, self.U_n))
        )

        h_t = (1.0 - u_t) * n_t + u_t * h_prev

        # Interpret hidden state as conditional variance η_t, enforce positivity via softplus
        return tf.nn.softplus(h_t)

    def forward(self, y: tf.Tensor, rv: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Run joint HN–GRU recursion and return standardized shocks and variances.

        Args
        -----
        y  : shape [T]
            Excess returns time series.
        rv : shape [T]
            Realized variance time series.

        Returns
        -------
        z_val : shape [T]
            Standardized shocks z_t = (y_t - λ η_t) / sqrt(η_t).
        h_val : shape [T]
            Variance path η_t used in the likelihood.
        """

        y = tf.convert_to_tensor(y, dtype=self.dtype)
        rv = tf.convert_to_tensor(rv, dtype=self.dtype)

        T = tf.shape(y)[0]
        z_array = tf.TensorArray(self.dtype, size=0, dynamic_size=True)
        h_array = tf.TensorArray(self.dtype, size=0, dynamic_size=True)

        # Initial variance: simple empirical variance of returns
        h_vol_t = tf.math.reduce_variance(y)
        eta_t = h_vol_t

        # t = 0 step
        z_t = (y[0] - self.lam * eta_t) / tf.sqrt(eta_t)
        z_array = z_array.write(0, z_t)
        h_array = h_array.write(0, h_vol_t)

        # Iterate for t = 1, ..., T-1
        for t in tf.range(1, T):

            # ==== 1. Build GRU input using previous eta, previous z, and current y, rv ====
            x_t = tf.stack([eta_t, y[t-1], rv[t-1], z_t])
            x_t = tf.reshape(x_t, (1, self.input_dim))

            # GRU dynamic bias uses previous h_vol_t
            eta_mat = self.gru_step(x_t, tf.reshape(h_vol_t, (1, 1)), tf.reshape(eta_t, (1, 1)))
            eta_t = tf.reshape(eta_mat, ())
            
            # ==== 3. Heston–Nandi update h_{t} -> h_{t+1} using PREVIOUS z_t ====
            h_vol_t = self.omega + self.phi * (h_vol_t - self.omega) + self.alpha * (tf.square(z_t) - 1.0 - 2.0 * self.gam * tf.sqrt(h_vol_t) * z_t)

            # ==== 2. Compute new standardized shock z_{t} using eta_t ====
            z_t = (y[t] - self.lam * eta_t) / tf.sqrt(eta_t)            

            # Record outputs
            z_array = z_array.write(t, z_t)
            h_array = h_array.write(t, h_vol_t)

        z_val = z_array.stack()
        h_val = h_array.stack()
        
        return z_val, h_val

    def neg_loglike(
        self,
        z_val: tf.Tensor,
        h_val: tf.Tensor,
    ) -> tf.Tensor:
        """Negative log-likelihood given standardized shocks and variances.

        We assume return dynamics of the form

            R_t = λ η_t + sqrt(η_t) z_t,   z_t ~ N(0, 1).

        The log-likelihood per observation is

            -1/2 [ log(2π) + log(η_t) + z_t^2 ].
        """

        z_val = tf.convert_to_tensor(z_val, dtype=self.dtype)
        h_val = tf.convert_to_tensor(h_val, dtype=self.dtype)

        log2pi = tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=self.dtype))
        nll = 0.5 * tf.reduce_sum(log2pi + tf.math.log(h_val) + tf.square(z_val))
        return nll


def train_epoch(
    model: VolIntegratedGRU,
    optimizer: tf.keras.optimizers.Optimizer,
    returns: tf.Tensor,
    rv: tf.Tensor,
) -> tf.Tensor:
    """One full-sequence gradient step (no batching) for SPX.

    This does MLE over the entire available history each call using the
    joint HN–GRU recursion.
    """

    with tf.GradientTape() as tape:
        z_val, h_val = model.forward(returns, rv)
        loss = model.neg_loglike(z_val, h_val)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


__all__ = ["VolIntegratedGRU", "train_epoch"]  # so if someone imports *