import numpy as np
import tensorflow as tf

class AdjointSolver:
    def __init__(self, eqn_config, probe_eps: float = 0.01,
                 use_jump: bool = True):
        self.delta     = float(eqn_config.discount)
        self.N         = int(eqn_config.num_time_interval_critic)
        self.T         = float(eqn_config.total_time_critic)
        self.dt        = self.T / self.N
        self.refl_tol  = 1e-8
        self.use_jump  = use_jump
        self.probe_eps = probe_eps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backward_pass(self, traj, dY, mask,
                      critic_model, phi_np, bsde):
        B, J, _ = traj.shape
        lam      = np.zeros((B, J), dtype=np.float64)   # λ(T) = 0
        lam_traj = np.zeros((B, J, self.N), dtype=np.float64)

        # Fixed system matrices
        B_eff   = tf.cast(bsde.B_mat, tf.float64)       # [J, K]
        B_eff_T = tf.transpose(B_eff)                   # [K, J]
        beta_tf = tf.cast(bsde.beta_v, tf.float64)      # [J]  base β (x-independent)
        c_tf    = tf.cast(bsde.c_weights, tf.float64)   # [K]
        u_lo    = tf.cast(bsde.u_lower, tf.float64)     # [K]
        u_hi    = tf.cast(bsde.u_upper, tf.float64)     # [K]

        # Critic conditioning: broadcast φ to all batch elements
        if phi_np is not None:
            phi_flat_np = phi_np.flatten()[np.newaxis, :]     # [1, K*J]
            phi_flat_tf = tf.constant(
                np.broadcast_to(phi_flat_np, (B, phi_flat_np.shape[1])),
                dtype=tf.float64)                             # [B, K*J]
        else:
            phi_flat_tf = None

        # ∇_x f = w (state-cost gradient, constant)
        df_x = np.tile(bsde.w_weights, (B, 1))           # [B, J]

        # Build the VJP function once and reuse across time steps
        vjp_fn = self._build_vjp_fn(critic_model, B_eff, B_eff_T,
                                    beta_tf, c_tf, u_lo, u_hi)

        # Backward integration: t = N−1, N−2, …, 0
        for t in reversed(range(self.N)):
            x_t1 = traj[:, :, t + 1]   # X̂(t_{n+1}): post-reflection state

            # ── Backward Euler step ──────────────────
            # λ̇(t+1) = δλ(t+1) − (∂_x b)^T λ(t+1) − w
            # λ(t)  ≈ λ(t+1) − dt · λ̇(t+1)
            rhs = self.delta * lam                        # δλ

            if phi_flat_tf is not None:
                # (∂_x b)^T λ = ∂/∂x [b(x)^T λ]   (β is x-independent, drops out)
                dxb_T_lam = vjp_fn(
                    tf.constant(x_t1, dtype=tf.float64),
                    tf.constant(lam,  dtype=tf.float64),
                    phi_flat_tf
                ).numpy()
                rhs = rhs - dxb_T_lam

            rhs  = rhs - df_x                            # subtract ∇_x f = w
            lam  = lam - self.dt * rhs                   # λ(t_n)

            # ── Jump correction─────
            if self.use_jump:
                lam = self._apply_jump_correction(
                    lam,
                    x_post=traj[:, :, t + 1],     # X̂(t_{n+1}): post-reflection
                    dY_t=dY[:, :, t],              # δY at step t
                    mask_t=mask[:, :, t],
                    critic_model=critic_model,
                    J=J,
                    phi_flat_tf=phi_flat_tf)

            lam_traj[:, :, t] = lam

        return lam_traj

    # ------------------------------------------------------------------
    # VJP: (∂_x b)^T λ
    # ------------------------------------------------------------------

    def _build_vjp_fn(self, critic_model, B_eff, B_eff_T,
                      beta_tf, c_tf, u_lo, u_hi):
        @tf.function(reduce_retracing=True)
        def _vjp_core(x_tf, lam_tf, phi_flat_tf):
            with tf.GradientTape() as tape:
                tape.watch(x_tf)
                # ∇_x V from the grad network (conditions on current φ)
                grad_V  = critic_model.NN_value_grad(
                    x_tf, phi_flat=phi_flat_tf, training=False)       # [B, J]
                # û*(x) = clip((B^T ∇V)_k / 2c_k)
                BT_grad = tf.linalg.matmul(grad_V, B_eff)             # [B, K]
                u_hat   = tf.clip_by_value(
                    BT_grad / (2.0 * c_tf), u_lo, u_hi)               # [B, K]
                # drift b = −B û* + β
                b       = -tf.linalg.matmul(u_hat, B_eff_T) + beta_tf # [B, J]
                # scalar for differentiation
                scalar  = tf.reduce_sum(b * lam_tf)
            grad = tape.gradient(scalar, x_tf)
            return grad if grad is not None else tf.zeros_like(x_tf)

        return _vjp_core

    # ------------------------------------------------------------------
    # Jump correction
    # ------------------------------------------------------------------

    def _apply_jump_correction(self, lam, x_post, dY_t, mask_t,
                               critic_model, J, phi_flat_tf=None):
        lam_out = lam.copy()

        # Find active reflection dimensions (any batch element reflects in dim j)
        active_dims = np.where(np.any(mask_t > 0.5, axis=0))[0]
        if len(active_dims) == 0:
            return lam_out

        # Collect boundary evaluation points across all active dimensions
        all_x   = []   # boundary point x with x_j = 0
        all_idx = []   # original batch indices
        all_dY  = []   # δY_j values for active samples
        all_j   = []   # which dimension this slice belongs to

        for j in active_dims:
            active = mask_t[:, j] > 0.5
            if not active.any():
                continue
            # Evaluate curvature at the boundary: x_j = 0, x_{-j} from post-step
            x_bnd = x_post[active].copy()
            x_bnd[:, j] = 0.0
            all_x.append(x_bnd)
            all_idx.append(np.where(active)[0])
            all_dY.append(dY_t[active, j])
            n = x_bnd.shape[0]
            all_j.append(np.full(n, j, dtype=np.int32))

        if not all_x:
            return lam_out

        # Concatenate for a single batched tape pass
        x_cat   = np.concatenate(all_x, axis=0)              # [N_active, J]
        j_cat   = np.concatenate(all_j, axis=0)               # [N_active]
        N_act   = x_cat.shape[0]

        # Broadcast φ to [N_active, K*J]
        if phi_flat_tf is not None:
            pf_tf = tf.broadcast_to(
                phi_flat_tf[:1],
                [N_act, phi_flat_tf.shape[1]])
        else:
            pf_tf = None

        onehot = tf.one_hot(j_cat, J, dtype=tf.float64)        # [N_active, J]

        x_tf = tf.constant(x_cat, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            G = critic_model.NN_value_grad(
                x_tf, phi_flat=pf_tf, training=False)          # [N_active, J]
            scalar = tf.reduce_sum(G * onehot)
        grad_x = tape.gradient(scalar, x_tf)                   # [N_active, J]

        if grad_x is None:
            return lam_out

        kappa_cat = tf.reduce_sum(grad_x * onehot, axis=1).numpy()   # [N_active]

        # Apply jump condition per dimension
        offset = 0
        for i, j in enumerate(active_dims):
            n     = all_x[i].shape[0]
            kappa = kappa_cat[offset:offset + n]
            idx   = all_idx[i]
            # λ_j(t_n) -= κ_j · δY_j(t_n)
            lam_out[idx, j] -= kappa * all_dY[i]
            offset += n
        return lam_out