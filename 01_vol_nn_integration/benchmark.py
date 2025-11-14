from __future__ import annotations
"""
Original structure for Heston-Nandi and Component Heston-Nandi
"""

import tensorflow as tf

class HestonNandi(tf.keras.Model):
    def __init__(self, dtype=tf.float32):
        super().__init__()
        self.dtype_ = dtype

        self.omega = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.000115867854986136),
            name="omega",
            dtype=dtype
        )
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(4.71105679172615e-06),
            name="alpha",
            dtype=dtype
        )
        self.phi = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.962822073703399),
            name="phi",
            dtype=dtype
        )
        self.lam = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(2.43378692806841),
            name="lam",
            dtype=dtype
        )
        self.gam = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(186.082260369241),
            name="gam",
            dtype=dtype
        )

    def forward(self, y):
        y = tf.convert_to_tensor(y, dtype=self.dtype_)
        T = tf.shape(y)[0]

        h = tf.TensorArray(self.dtype_, size=T)
        h_t = tf.math.reduce_variance(y)
        h = h.write(0, h_t)

        for t in tf.range(1, T):
            z_prev = (y[t-1] - self.lam * h_t) / tf.sqrt(h_t)
            h_t = (
                self.omega
                + self.phi * (h_t - self.omega)
                + self.alpha * (
                    tf.square(z_prev)
                    - 1.0
                    - 2.0 * self.gam * tf.sqrt(h_t) * z_prev
                )
            )
            h = h.write(t, h_t)

        return h.stack()


class ComponentHestonNandi(tf.keras.Model):
    def __init__(self, dtype=tf.float32):
        super().__init__()
        self.dtype_ = dtype

        self.omega = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.000115944368734123),
            name="omega",
            dtype=dtype
        )
        self.alpha = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(2.91880490811086e-06),
            name="alpha",
            dtype=dtype
        )
        self.phi = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.887721679393748),
            name="phi",
            dtype=dtype
        )
        self.lam = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(2.38788661600365),
            name="lam",
            dtype=dtype
        )
        self.gam1 = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(332.934927419811),
            name="gam1",
            dtype=dtype
        )
        self.gam2 = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(134.424516082664),
            name="gam2",
            dtype=dtype
        )
        self.vphi = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(1.61007979938689e-06),
            name="vphi",
            dtype=dtype
        )
        self.rho = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(0.990486709209338),
            name="rho",
            dtype=dtype
        )

    def forward(self, y):
        y = tf.convert_to_tensor(y, dtype=self.dtype_)
        T = tf.shape(y)[0]

        h = tf.TensorArray(self.dtype_, size=T)
        q_t = tf.math.reduce_variance(y)
        h_t = q_t
        h = h.write(0, h_t)

        for t in tf.range(1, T):
            z_prev = (y[t-1] - self.lam * h_t) / tf.sqrt(h_t)

            # ---- q_{t+1} update (Eq. 7) ----
            q_t = (
                self.omega
                + self.rho * (q_t - self.omega)
                + self.vphi * (
                    tf.square(z_prev)
                    - 1.0
                    - 2.0 * self.gam2 * tf.sqrt(h_t) * z_prev
                )
            )

            # ---- h_{t+1} update (Eq. 6) ----
            h_t = (
                q_t
                + self.phi * (h_t - q_t)
                + self.alpha * (
                    tf.square(z_prev)
                    - 1.0
                    - 2.0 * self.gam1 * tf.sqrt(h_t) * z_prev
                )
            )

            h = h.write(t, h_t)

        return h.stack()


__all__ = ["HestonNandi", "ComponentHestonNandi"]
