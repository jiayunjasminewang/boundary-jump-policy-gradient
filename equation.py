import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import build_sigma


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class Equation:
    def __init__(self, eqn_config):
        self.dim   = eqn_config.dim
        self.gamma = eqn_config.discount   # δ > 0 → discounted; ≈ 0 → ergodic
        self.x0    = eqn_config.x0
        self.rho   = getattr(eqn_config, 'rho', 0.0)
        self.a     = getattr(eqn_config, 'a', 1)

        R_cfg = getattr(eqn_config, 'R', 0.0)
        if (not isinstance(R_cfg, list)) and abs(R_cfg) < 1e-9 and abs(self.rho) < 1e-9:
            self.rep_flag = True
        else:
            self.rep_flag = False

    # ------------------------------------------------------------------
    # Sample generation  (returns y, dw, dY)
    # ------------------------------------------------------------------

    def gen_samples(self, num_sample, T, N, Total_iterations, simulation_method):
        delta_t      = T / N
        sqrt_delta_t = np.sqrt(delta_t)
        R            = self.R

        total_steps = N * Total_iterations

        y_sample  = np.empty((num_sample, self.dim, total_steps + Total_iterations), dtype=np.float64)
        dw_sample = np.empty((num_sample, self.dim, total_steps), dtype=np.float64)
        dY_sample = np.empty((num_sample, self.dim, total_steps), dtype=np.float64)

        x0  = np.ones(self.dim) * self.x0
        y_i = np.tile(x0, num_sample).reshape([num_sample, self.dim])

        y_col   = 0
        step_col = 0

        for i in tqdm(range(total_steps),
                      desc='Generating samples', leave=False):

            if i % N == 0:
                if simulation_method == 'fixed':
                    y_i = np.tile(x0, num_sample).reshape([num_sample, self.dim])
                elif simulation_method == 'uniform':
                    y_i = np.random.uniform(size=(num_sample, self.dim)) * 2.0
                y_sample[:, :, y_col] = y_i
                y_col += 1

            
            dw_sigma = self.diffusion(
                np.random.normal(size=(num_sample, self.dim))
            ) * sqrt_delta_t
            dw_sample[:, :, step_col] = dw_sigma

            x_free = y_i + self.drift_np(y_i) * delta_t + dw_sigma
            y_i, dY_i = self.skorokhod_with_increment(x_free, R)

            dY_sample[:, :, step_col] = dY_i
            y_sample[:, :, y_col]     = y_i
            y_col    += 1
            step_col += 1

        return y_sample, dw_sample, dY_sample

    # ------------------------------------------------------------------
    # Skorokhod map  (returns reflected state and increment)
    # ------------------------------------------------------------------

    @staticmethod
    def skorokhod_with_increment(X_full, R):
        """
        Project X_full onto R^d_+ using the Skorokhod map with reflection
        matrix R.  Returns:
            Y   : reflected state   [num_sample, dim]
            dY  : pushing increment [num_sample, dim]  (≥ 0, elementwise)
        """
        eps         = 1e-8
        num_samples = X_full.shape[0]
        Y   = np.zeros_like(X_full)
        dY  = np.zeros_like(X_full)

        for i in range(num_samples):
            x = X_full[i, :].copy()
            y = x.copy()
            l = np.zeros_like(x)   # cumulative local time

            while np.any(y < -eps):
                active = y < eps
                R_bb = R[np.ix_(active, active)]
                if R_bb.size == 0:
                    break
                l_b = -np.linalg.solve(R_bb, x[active])
                y   = x + R[:, active] @ l_b
                l[active] = l_b

            Y[i,  :] = y
            dY[i, :] = np.maximum(l, 0.0)

        return Y, dY

    # ------------------------------------------------------------------
    # Overridable methods
    # ------------------------------------------------------------------

    def drift_np(self, x):
        """Drift vector as numpy array (reference / no-control drift)."""
        raise NotImplementedError

    def diffusion(self, dw):
        """Apply diffusion matrix to Brownian increments."""
        raise NotImplementedError

    def w_tf(self, x, grad_V, a_lowbound=0.0):
        """Running cost for the BSDE loss (TF tensors)."""
        raise NotImplementedError

    def V_true(self, x):
        return None

    def V_grad_true(self, x):
        return None

class DynamicPricing(Equation):
    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.name = 'dynamicPricing'

        R = np.identity(self.dim)
        q_ex = getattr(eqn_config, 'queue_example', '')
        if q_ex == '2':
            R[0, 1:] = eqn_config.R
        else:
            R[1:, 0] = eqn_config.R
        self.R  = R
        self.mu = np.repeat(eqn_config.mu, self.dim)

        if self.rep_flag:
            self.h = np.repeat(eqn_config.h, self.dim)
        else:
            if q_ex == '2':
                self.h    = np.repeat(eqn_config.h, self.dim)
                self.h[0] = eqn_config.h * 0.95
            else:
                self.h    = np.repeat(eqn_config.h * 0.95, self.dim)
                self.h[0] = eqn_config.h

        if hasattr(eqn_config, 'cTrue'):
            self.cTrue = eqn_config.cTrue
        if hasattr(eqn_config, 'constOptimal'):
            self.constOpt = eqn_config.constOptimal
        if hasattr(eqn_config, 'linearOptimal'):
            self.linearOpt = eqn_config.linearOptimal

    def drift_np(self, x):
        return np.tile(self.mu, (x.shape[0], 1))

    def diffusion(self, dw):
        mat = np.identity(self.dim)
        for i in range(1, self.dim):
            for j in range(1, self.dim):
                if i != j:
                    mat[i, j] = self.rho
        L = np.linalg.cholesky(mat)
        return np.dot(L, dw.T).T

    def w_tf(self, x, grad_V, a_lowbound=0.0):
        w = tf.linalg.matvec(x, tf.cast(self.h, x.dtype))
        w = tf.reshape(w, [-1, 1])
        w = w - tf.reduce_sum(tf.square(grad_V), axis=1, keepdims=True) / 4.0
        return w, tf.constant(0.0, dtype=x.dtype)


class ThinStream(Equation):
    """
    Linear cost (thin stream / input control) drift control.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.name = 'thinStream'

        R = np.identity(self.dim)
        q_ex = getattr(eqn_config, 'queue_example', '')
        if q_ex == '2':
            R[0, 1:] = eqn_config.R
        else:
            R[1:, 0] = eqn_config.R
        self.R  = R
        self.mu = np.repeat(eqn_config.mu, self.dim)
        self.v  = np.repeat(eqn_config.v,  self.dim)

        if self.rep_flag:
            self.h = np.repeat(eqn_config.h, self.dim)
        else:
            if q_ex == '2':
                self.h    = np.repeat(eqn_config.h, self.dim)
                self.h[0] = eqn_config.h * 0.95
            else:
                self.h    = np.repeat(eqn_config.h * 0.95, self.dim)
                self.h[0] = eqn_config.h

        if hasattr(eqn_config, 'cTrue'):
            self.cTrue = eqn_config.cTrue

    def drift_np(self, x):
        return np.tile(self.mu, (x.shape[0], 1))

    def diffusion(self, dw):
        mat = np.identity(self.dim)
        if self.rho > 1.0:
            for i in range(1, self.dim):
                for j in range(1, self.dim):
                    if i != j:
                        mat[i, j] = -self.R[i, 0] * self.R[j, 0]
        else:
            for i in range(1, self.dim):
                for j in range(1, self.dim):
                    if i != j:
                        mat[i, j] = self.rho
        L = np.linalg.cholesky(mat)
        return np.dot(L, dw.T).T

    def w_tf(self, x, grad_V, a_lowbound=0.0):
        h_tf = tf.cast(self.h, x.dtype)
        mu_tf = tf.cast(self.mu, x.dtype)
        v_tf  = tf.cast(self.v,  x.dtype)

        # h^T x  (holding cost)
        w = tf.linalg.matvec(x, h_tf)
        # − μ^T ∇V  (drift contribution to HJB running cost)
        w = w - tf.linalg.matvec(grad_V, mu_tf)
        w = tf.reshape(w, [-1, 1])

        # bang-bang control term: split into max and min parts
        grad_minus_v   = grad_V - v_tf[None, :]
        max_zero_grad  = tf.math.maximum(0., grad_minus_v)   # positive part
        min_zero_grad  = tf.math.minimum(0., grad_minus_v)   # negative part

        w = w - tf.reduce_sum(max_zero_grad, axis=1, keepdims=True) \
                * tf.cast(self.a, x.dtype)
        w = w - tf.reduce_sum(min_zero_grad, axis=1, keepdims=True) \
                * tf.cast(a_lowbound, x.dtype)

        return w, tf.constant(0.0, dtype=x.dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Multiclass Queue under Halfin-Whitt scaling
# ──────────────────────────────────────────────────────────────────────────────

class MulticlassQueue(Equation):

    def __init__(self, eqn_config):
        super().__init__(eqn_config)
        self.name = 'multiclassQueue'
        # ── dimensions ────────────────────────────────────────────────
        self.K = eqn_config.K

        if hasattr(eqn_config, 'M_cap'):
            self.M_cap    = np.array(eqn_config.M_cap,  dtype=np.float64).T  # [K,J] → [J,K]
            self.C_cap    = self.M_cap                                        # [J, K]
            self.beta_v   = np.array(eqn_config.beta,   dtype=np.float64)  # [J]
            if hasattr(eqn_config, 'phi_star'):
                self.phi_star = np.array(eqn_config.phi_star, dtype=np.float64)
            else:
                self.phi_star = np.full((self.K, self.dim), 1.0 / self.K, dtype=np.float64)                       # [K, J] uniform default
            self.theta    = self.beta_v.copy()
        else:
            self.theta    = np.array(eqn_config.theta,    dtype=np.float64)
            self.mu_kj    = np.array(eqn_config.mu_kj,    dtype=np.float64)  # [K, J]
            self.alpha_k  = np.array(eqn_config.alpha_k,  dtype=np.float64)  # [K]
            self.mu_star  = np.array(eqn_config.mu_star,  dtype=np.float64)  # [K]
            self.phi_star = np.array(eqn_config.phi_star, dtype=np.float64)  # [K, J]
            self.M_cap  = self.mu_kj.T * (self.alpha_k * self.mu_star)[np.newaxis, :]  # [J, K]
            self.C_cap  = self.mu_kj.T * self.alpha_k[np.newaxis, :]                  # [J, K]
            self.beta_v = self.theta.copy()

        self.B_mat = self.C_cap * self.phi_star.T   # [J, K]

        # ── diffusion covariance Σ ────────────────────────────────────
        if hasattr(eqn_config, 'M_cap'):
            self.Sigma = build_sigma(eqn_config.Sigma, self.dim)
        else:
            Lambda = np.einsum('kj,k,kj,k->j',
                               self.mu_kj, self.alpha_k,
                               self.phi_star, self.mu_star)          # [J]
            Sigma_derived = np.diag(Lambda)
            for k in range(self.K):
                mu_phi = self.mu_kj[k, :] * self.phi_star[k, :]     # [J]
                Sigma_derived += (self.alpha_k[k] * self.mu_star[k]
                                  * np.outer(mu_phi, mu_phi))
            self.Sigma = Sigma_derived

            if hasattr(eqn_config, 'Sigma'):
                import logging
                cfg_sig = build_sigma(eqn_config.Sigma, self.dim)
                if not np.allclose(cfg_sig, self.Sigma, rtol=0.05):
                    logging.warning(
                        'queue Σ: config value differs from eq.3.10 derivation '
                        '(ignored).\n  config:  %s\n  derived: %s',
                        cfg_sig.tolist(), self.Sigma.tolist())

        self.Sigma_sqrt = np.linalg.cholesky(self.Sigma)

        # ── cost parameters ───────────────────────────────────────────
        self.w_weights = np.array(eqn_config.w,       dtype=np.float64)  # [J]
        self.c_weights = np.array(eqn_config.c,       dtype=np.float64)  # [K]
        self.u_lower   = np.array(eqn_config.u_lower, dtype=np.float64)  # [K]
        self.u_upper   = np.array(eqn_config.u_upper, dtype=np.float64)  # [K]

        # ── initial routing (uniform) ──────────────────────────────────
        if hasattr(eqn_config, 'phi_init'):
            phi0 = np.array(eqn_config.phi_init, dtype=np.float64)  # [K, J]
        else:
            phi0 = np.full((self.K, self.dim), 1.0 / self.K)
        self.phi_init = phi0  # stored for reference; Actor will optimise from here

        # ── orthogonal reflection onto R^J_+ ─────────────────────────
        self.R = np.eye(self.dim)

        # ── TF constants ──────────────────────────────────────────────
        self._beta_tf  = tf.constant(self.beta_v,   dtype=tf.float64)   # [J]
        self._w_tf     = tf.constant(self.w_weights, dtype=tf.float64)  # [J]
        self._c_tf     = tf.constant(self.c_weights, dtype=tf.float64)  # [K]

        # compatibility fields used by Equation base / solver
        self.a  = getattr(eqn_config, 'a', 1)
        self.mu = np.zeros(self.dim)

    # ------------------------------------------------------------------
    # Effective B matrix given routing fractions φ
    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def drift_np(self, x):
        """
        Reference drift used during sample generation.
        û=0 during sample gen, so drift = β  (paper eq. 3.13 with û=0).
        """
        B_np = x.shape[0]
        return np.tile(self.beta_v, (B_np, 1))               # [B, J]

    def drift_with_phi_uhat(self, x_np, phi_np, uhat_np):
        """
        Full controlled drift: b = -B_eff · û + β  (paper eq. 3.13).
        phi_np  : [K, J]
        uhat_np : [B, K]
        returns : [B, J]
        """
        B_eff = self.compute_B_eff_np(phi_np)                  # [J, K]
        return -(uhat_np @ B_eff.T) + self.beta_v[None, :]    # [B, J]

    def diffusion(self, dw):
        return (self.Sigma_sqrt @ dw.T).T

    # ------------------------------------------------------------------
    # Optimal online intensity control
    # ------------------------------------------------------------------

    def compute_B_eff_np(self, phi_np):
        """B_eff[j,k] = C_cap[j,k] · φ_kj = μ_kj · α_k · φ_kj  (paper eq. 3.12).
        phi_np: [K, J] → returns [J, K]."""
        return self.C_cap * phi_np.T   # [J, K]

    def optimal_u_hat_np(self, grad_V_np, phi_np=None):
        B = self.compute_B_eff_np(phi_np) if phi_np is not None else self.B_mat
        raw = (grad_V_np @ B) / (2.0 * self.c_weights)
        return np.clip(raw, self.u_lower, self.u_upper)

    def optimal_u_hat_tf(self, grad_V_tf):
        B_tf    = tf.cast(self.B_mat, grad_V_tf.dtype)
        c_tf    = tf.cast(self._c_tf, grad_V_tf.dtype)
        BT_grad = tf.linalg.matmul(grad_V_tf, B_tf)
        raw     = BT_grad / (2.0 * c_tf)
        return tf.clip_by_value(raw,
                                tf.cast(self.u_lower, raw.dtype),
                                tf.cast(self.u_upper, raw.dtype))

    # ------------------------------------------------------------------
    # Running cost  (for Critic BSDE loss)
    # ------------------------------------------------------------------

    def w_tf(self, x, grad_V, a_lowbound=0.0, B_eff_tf=None):
        """
        f(x, û*(x;φ)) = w^T x  −  Σ_k (B_eff^T ∇V)_k² / (4 c_k)

        B_eff_tf: [J, K] dynamic matrix for solution operator (φ-conditioned).
                  If None, falls back to fixed self.B_mat.
        """
        w_tf_c = tf.cast(self._w_tf, x.dtype)
        c_tf   = tf.cast(self._c_tf, x.dtype)
        B_tf   = tf.cast(B_eff_tf, x.dtype) if B_eff_tf is not None \
                 else tf.cast(self.B_mat, x.dtype)          # [J, K]

        state_cost = tf.linalg.matvec(x, w_tf_c)           # [B]
        BT_grad    = tf.linalg.matmul(grad_V, B_tf)        # [B, K]
        ctrl_gain  = tf.reduce_sum(
            tf.square(BT_grad) / (4.0 * c_tf), axis=1)

        running = tf.reshape(state_cost - ctrl_gain, [-1, 1])
        return running, tf.constant(0.0, dtype=x.dtype)

    # ------------------------------------------------------------------
    # Numpy running cost for evaluation
    # ------------------------------------------------------------------

    def running_cost_np(self, x_np, uhat_np):
        state = x_np   @ self.w_weights
        ctrl  = uhat_np**2 @ self.c_weights
        return float(np.mean(state + ctrl))

    def beta_from_hat_phi(self, hat_phi):
        """hat_phi: np.ndarray [K, J] → corrected β [J].
        Uses M_cap directly: correction[j] = Σ_k M_cap[j,k] * hat_phi[k,j]
        """
        return self.theta - np.einsum('jk,kj->j', self.M_cap, hat_phi)

# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY = {
    'dynamicPricing'  : DynamicPricing,
    'thinStream'      : ThinStream,
    'multiclassQueue' : MulticlassQueue,
}


def get_equation(eqn_config):
    name = eqn_config.eqn_name
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown equation '{name}'. "
            f"Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](eqn_config)