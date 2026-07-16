import logging
import os
import time

import numpy as np
import tensorflow as tf

from adjoint import AdjointSolver
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Neural network building block
# ──────────────────────────────────────────────────────────────────────────────

class DeepNN(tf.keras.Model):
    def __init__(self, config, mode: str):
        super().__init__()
        self.mode       = mode
        self.activation = config.net_config.activation
        self.eqn_config = config.eqn_config

        dim = config.eqn_config.dim
        K   = getattr(config.eqn_config, 'K', dim)

        num_hiddens = config.net_config.num_hiddens_critic

        self.dense_layers = [
            tf.keras.layers.Dense(h, use_bias=False, activation=None,
                                  dtype=tf.float64)
            for h in num_hiddens
        ]

        if mode == 'critic':
            self.dense_layers.append(
                tf.keras.layers.Dense(1, activation=None, use_bias=True,
                                      dtype=tf.float64))
        elif mode == 'critic_grad':
            self.dense_layers.append(
                tf.keras.layers.Dense(dim, activation=None, use_bias=False,
                                      dtype=tf.float64))
            KJ      = K * dim
            enc_dim = getattr(config.net_config, 'phi_enc_hidden', 16)
            self.phi_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(enc_dim, use_bias=True, dtype=tf.float64,
                                      activation='elu'),
                tf.keras.layers.Dense(enc_dim, use_bias=True, dtype=tf.float64,
                                      activation='elu'),
            ])
        elif mode == 'actor':
            # output K*J logits; ActorModel applies softmax to get [B, K, J]
            self.dense_layers.append(
                tf.keras.layers.Dense(K * dim, activation=None, use_bias=True,
                                      dtype=tf.float64))
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def call(self, x, phi_flat=None, training=False):
        # ── Neumann BC: critic value network parameterized through x² ─
        # ── Solution operator: φ concatenated after x² transform      ─
        if self.mode == 'critic':
            z = tf.square(x)
            y = tf.concat([z, tf.cast(phi_flat, z.dtype)], axis=1) if phi_flat is not None else z
        elif self.mode == 'critic_grad':
            x = tf.cast(x, tf.float64)
            z = tf.square(x)
            if phi_flat is not None:
                phi_emb = self.phi_encoder(tf.cast(phi_flat, tf.float64))
                y = tf.concat([z, phi_emb], axis=1)
            else:
                y = z
        else:
            y = x

        for layer in self.dense_layers[:-1]:
            y = layer(y)
            if self.activation == 'relu':
                y = tf.nn.relu(y)
            elif self.activation == 'leaky':
                y = tf.nn.leaky_relu(y)
            elif self.activation == 'elu':
                y = tf.nn.elu(y)
            elif self.activation == 'sigmoid':
                y = y + tf.math.sigmoid(y) - 0.5

        y = self.dense_layers[-1](y)
        if self.mode == 'critic_grad':
            return 2.0 * x * y   # x was cast to float64 above; G_j = 2x_j · G̃_j
        return y

# ──────────────────────────────────────────────────────────────────────────────
# Critic model
# ──────────────────────────────────────────────────────────────────────────────

class CriticModel(tf.keras.Model):
    """
    Double-parametrization Critic (from RBMSolver).
    Two networks: NN_value (V_w) and NN_value_grad (G_w ≈ ∇V_w).
    """

    def __init__(self, config, bsde):
        super().__init__()
        self.eqn_config  = config.eqn_config
        self.net_config  = config.net_config
        self.bsde        = bsde
        self.gamma       = config.eqn_config.discount
        self.steady_flag = (self.gamma < 1e-9)

        self.NN_value      = DeepNN(config, 'critic')
        self.NN_value_grad = DeepNN(config, 'critic_grad')

        self.control = hasattr(config.train_config, 'control')

    def call(self, inputs, training=False):
        """Keras requires call(); delegates to forward() which does the real work."""
        return self.forward(inputs, training=training, use_NN=True)

    def forward(self, inputs, training=False, use_NN=True, phi_flat=None):
        """
        Evaluate the BSDE stochastic identity and return:
            delta   : residual  [B, 1]
            cur_val : current estimate of V(x0)  (scalar)
        phi_flat : tf.Tensor [B, K*J] or None.
          If provided, conditions both NN_value and NN_value_grad on φ
          (solution operator mode). If None, uses fixed B_mat (baseline mode).
        """
        a_lowbound, dw, y_sample = inputs

        dw       = tf.cast(dw,       tf.float64)
        y_sample = tf.cast(y_sample, tf.float64)

        N  = self.eqn_config.num_time_interval_critic
        dt = self.eqn_config.total_time_critic / N

        B_eff_tf = tf.cast(self.bsde.B_mat, tf.float64) if hasattr(self.bsde, 'B_mat') else None

        if use_NN:

            d = tf.shape(dw)[1]

            # y_sample: [B, d, N+1] → x_all: [B, N, d]
            x_all  = tf.transpose(y_sample[:, :, :N], [0, 2, 1])      # [B, N, d]
            dw_all = tf.transpose(dw,                 [0, 2, 1])       # [B, N, d]
            x_flat  = tf.reshape(x_all,  [-1, d])                      # [B*N, d]
            dw_flat = tf.reshape(dw_all, [-1, d])                      # [B*N, d]

            # Expand phi_flat: [B, K*J] → [B*N, K*J]
            if phi_flat is not None:
                phi_flat_rep = tf.repeat(phi_flat, N, axis=0)          # [B*N, K*J]
            else:
                phi_flat_rep = None

            # Single batched NN forward pass
            grad_flat = self.NN_value_grad(x_flat, phi_flat=phi_flat_rep,
                                           training=training)           # [B*N, d]

            # Running costs for all (b, t) at once
            if self.control:
                w_flat, _ = self.bsde.w_tf(x_flat, grad_flat, a_lowbound,
                                            B_eff_tf=B_eff_tf)  # [B*N,1]
            else:
                w_flat, _ = self.bsde.w_tf(x_flat)                    # [B*N, 1]

            # Reshape back to [B, N, *]
            B_dyn    = tf.shape(dw)[0]
            grad_all = tf.reshape(grad_flat, [B_dyn, N, d])            # [B, N, d]
            w_all    = tf.reshape(w_flat,    [B_dyn, N, 1])            # [B, N, 1]

            # Diffusion integral: G_w · dW  →  [B, N, 1]
            dw_dot_all = tf.reduce_sum(
                tf.reshape(dw_flat, [B_dyn, N, d]) * grad_all,
                axis=-1, keepdims=True)                                 # [B, N, 1]

            # Precomputed discount vector: exp(-γ·t·dt) for t=0…N-1
            t_idx    = tf.cast(tf.range(N), tf.float64)
            disc_vec = tf.exp(-tf.cast(self.gamma * dt, tf.float64) * t_idx)
            disc_vec = tf.reshape(disc_vec, [1, N, 1])                 # [1, N, 1]

            # y[b] = Σ_t  disc_t · (w_t · dt − dw_dot_t)
            y = tf.reduce_sum(disc_vec * (w_all * dt - dw_dot_all),
                              axis=1)                                   # [B, 1]

            # Terminal discount  e^{-γ·T}
            discount = tf.exp(tf.cast(-self.gamma * N * dt, tf.float64))

        else:
            # Non-NN path (evaluation / reference)
            y        = tf.zeros([tf.shape(dw)[0], 1], dtype=tf.float64)
            discount = tf.constant(1.0, dtype=tf.float64)

            for t in range(N):
                x_t  = y_sample[:, :, t]
                x_tf = tf.Variable(x_t, dtype=tf.float64)
                with tf.GradientTape() as tape:
                    v_t = self.NN_value(x_tf, phi_flat=phi_flat, training=training)
                grad_t = tape.gradient(v_t, x_tf)

                if self.control:
                    w_t, _ = self.bsde.w_tf(
                        x_t, self.bsde.V_grad_true(x_t), a_lowbound)
                else:
                    w_t, _ = self.bsde.w_tf(x_t)

                y = y + discount * w_t * dt

                dw_t  = dw[:, :, t]
                dw_dot = tf.reduce_sum(
                    dw_t * self.bsde.V_grad_true(x_t), axis=1, keepdims=True)
                y = y - discount * dw_dot

                discount = discount * tf.exp(
                    tf.cast(-self.gamma * dt, tf.float64))

        # Terminal residual:  δ = V(x_0) − y − V(x_T) · e^{-γT}
        x0_batch = y_sample[:, :, 0]
        xT_batch = y_sample[:, :, -1]

        if use_NN:
            V0 = self.NN_value(x0_batch, phi_flat=phi_flat, training=training)
            VT = self.NN_value(xT_batch, phi_flat=phi_flat, training=training)
        else:
            V0 = self.bsde.V_true(x0_batch)
            VT = self.bsde.V_true(xT_batch)

        delta = V0 - y - VT * discount

        if self.steady_flag:
            cur_val = -tf.reduce_mean(delta) / self.eqn_config.total_time_critic
        else:
            cur_val = -tf.reduce_mean(delta) / (
                1.0 - tf.exp(tf.cast(
                    -self.gamma * self.eqn_config.total_time_critic,
                    tf.float64)))

        return delta, cur_val


# ──────────────────────────────────────────────────────────────────────────────
# Actor model
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────

class ControlSolver:

    def __init__(self, config, bsde):
        self.eqn_config  = config.eqn_config
        self.net_config  = config.net_config
        self.train_config = config.train_config
        self.bsde        = bsde

        self.gamma       = config.eqn_config.discount
        self.steady_flag = (self.gamma < 1e-9)

        self.loss_rec  = []
        self.val_rec   = []
        self.actor_loss_rec = []

        self.true_ld    = getattr(config.eqn_config, 'alow', 0.0)
        self.queue_example = getattr(config.eqn_config, 'queue_example', '')

        # ── Critic ────────────────────────────────────────────────────
        if hasattr(config.train_config, 'control'):
            self.control = True

        self.model_critic = CriticModel(config, bsde)
        lr_critic = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            config.net_config.lr_boundaries_critic,
            config.net_config.lr_values_critic)
        self.optimizer_critic = tf.keras.optimizers.Adam(
            learning_rate=lr_critic, epsilon=1e-8)

        # ── Actor (for MulticlassQueue) ───────────────────────────────
        self.has_actor = (bsde.name == 'multiclassQueue')
        if self.has_actor:
            K = bsde.K
            J = bsde.dim
            self.phi_logits = tf.Variable(
                np.zeros((K, J), dtype=np.float64), trainable=True, dtype=tf.float64)
            if hasattr(self.eqn_config, 'phi_mask'):
                adj = np.array(self.eqn_config.phi_mask, dtype=np.float64)
            elif hasattr(bsde, 'M_cap'):
                adj = (np.array(bsde.M_cap, dtype=np.float64).T > 1e-8).astype(float)
            else:
                adj = np.ones((K, J), dtype=np.float64)

            valid_per_pool = adj.sum(axis=1)
            assert np.all(valid_per_pool > 0), \
                f"Pool(s) {np.where(valid_per_pool == 0)[0]} have no valid routing targets"

            self.phi_logit_mask = tf.constant(
                np.where(adj == 0, -1e9, 0.0), dtype=tf.float64)
            lr_actor = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                config.net_config.lr_boundaries_actor,
                config.net_config.lr_values_actor)
            self.optimizer_actor = tf.keras.optimizers.Adam(
                learning_rate=lr_actor, epsilon=1e-8)
            self.actor_update_freq = getattr(
                config.train_config, 'actor_update_freq', 10)
            self.actor_warmup_steps = getattr(
                config.train_config, 'actor_warmup_steps', 0)
            # Dirichlet concentration for φ sampling (solution operator training)
            self.phi_sample_alpha  = getattr(
                config.train_config, 'phi_sample_alpha', 1.0)
            probe_eps = getattr(config.train_config, 'probe_eps', 0.01)
            use_jump  = getattr(config.train_config, 'use_jump',  True)
            self.adjoint_solver = AdjointSolver(config.eqn_config,
                                                probe_eps=probe_eps,
                                                use_jump=use_jump)
            # ── [FIX] φ-logit clip range, now configurable ─────────────────
            # Root cause found via diagnostics: with the old hardcoded
            # [-1.5, 1.5], jump=True and jump=False both saturate to the
            # SAME logit values (±1.5), so the two runs converge to an
            # identical φ regardless of any real difference in λ upstream
            # — the clip, not the physics, was determining the answer.
            # Default widened to [-6.0, 6.0] (still bounds the softmax
            # Jacobian away from total numerical saturation). Set
            # train_config.phi_logit_clip_range = null in the JSON config
            # to disable clipping entirely.
            clip_cfg = getattr(config.train_config,
                               'phi_logit_clip_range', [-6.0, 6.0])
            self._phi_logit_clip = (tuple(clip_cfg)
                                    if clip_cfg is not None else None)
        else:
            self.adjoint_solver = None


    # ------------------------------------------------------------------
    # Sample generation / loading
    # ------------------------------------------------------------------

    def gen_samples(self, dump: bool = True, load: bool = False):
        start = time.time()
        B   = self.net_config.batch_size
        T   = self.eqn_config.total_time_critic
        N   = self.eqn_config.num_time_interval_critic
        itr = self.net_config.num_iterations

        sim = getattr(self.eqn_config, 'simulation', '')

        logging.info('dim=%d  γ=%.3f  T=%.2f  N=%d  B=%d',
                     self.bsde.dim, self.gamma, T, N, B)

        fname = self._sample_filename(sim)

        if load:
            self.dw_sample  = np.load(fname + '_w.npy',  allow_pickle=True)
            self.y_sample   = np.load(fname + '_y.npy',  allow_pickle=True)
            self.dY_sample  = np.load(fname + '_dY.npy', allow_pickle=True)
        else:
            (self.y_sample,
             self.dw_sample,
             self.dY_sample) = self.bsde.gen_samples(B, T, N, itr + 20, sim)

            if dump:
                os.makedirs('data', exist_ok=True)
                np.save(fname + '_w',  self.dw_sample)
                np.save(fname + '_y',  self.y_sample)
                np.save(fname + '_dY', self.dY_sample)

        # Reflection mask: 1 where ΔY > tolerance
        self.mask_sample = (
            self.dY_sample > getattr(self.train_config, 'refl_tol', 1e-8)
        ).astype(np.float64)

        logging.info('Samples ready. shape=%s  time=%.1fs',
                     str(self.y_sample.shape), time.time() - start)

    def _sample_filename(self, sim):
        B   = self.net_config.batch_size
        T   = self.eqn_config.total_time_critic
        N   = self.eqn_config.num_time_interval_critic
        itr = getattr(self.net_config, 'sample_iterations',
                      self.net_config.num_iterations)
        return (f"data/{self.bsde.name}_{B}_{T}_{N}_{itr}"
                f"_dim={self.bsde.dim}{sim}")

    def load_critic_weights(self, ckpt_value, ckpt_grad):
        B   = self.net_config.batch_size
        J   = self.bsde.dim
        K   = self.bsde.K
        dummy_x      = np.zeros((1, J),   dtype=np.float64)
        dummy_phi    = np.zeros((1, K*J), dtype=np.float64)
        self.model_critic.NN_value(dummy_x,      phi_flat=tf.constant(dummy_phi, dtype=tf.float64))
        self.model_critic.NN_value_grad(dummy_x, phi_flat=tf.constant(dummy_phi, dtype=tf.float64))

        self.model_critic.NN_value.load_weights(ckpt_value)
        self.model_critic.NN_value_grad.load_weights(ckpt_grad)
        logging.info('Critic weights loaded from %s / %s', ckpt_value, ckpt_grad)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        start    = time.time()
        B        = self.net_config.batch_size
        T        = self.eqn_config.total_time_critic
        N        = self.eqn_config.num_time_interval_critic
        itr      = self.net_config.num_iterations
        sim      = getattr(self.eqn_config, 'simulation', '')

        # Epoch count (mirrors RBMSolver heuristic)
        if self.bsde.dim <= 10:
            epoch_total = self.bsde.dim * 2 + self.bsde.a // 2 + 10
        else:
            epoch_total = self.bsde.dim     + self.bsde.a // 2 + 10
        epoch_total = getattr(self.train_config, 'epoch', epoch_total)
        logging.info('Epoch total: %d', epoch_total)

        # Build validation data (last 20 episodes of sample buffer)
        valid_w, valid_y = self._build_validation_data(B, N, itr)
        valid_data = (0.0, valid_w, valid_y)

        # Pace for thinStream lower-bound annealing
        pace = getattr(self.train_config, 'pace', 0)

        # Output directories
        act_str = self.net_config.activation
        os.makedirs(f'figs_{act_str}', exist_ok=True)
        os.makedirs(f'logs_{act_str}', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        suffix = self._gen_filename()
        logging.info('Run name: %s', suffix)

        global_step = 0

        for epoch in range(epoch_total):
            logging.info('Epoch %d / %d', epoch, epoch_total)
            discard = 1000 if epoch > 0 else 0

            for step in range(itr):
                if epoch > 0 and step < discard:
                    global_step += 1
                    continue

                cs, ce   = step * N,     (step + 1) * N          # dw : N entries
                csy, cey = step * N,     step * N + N + 1        # y  : N+1 entries

                # lower-bound annealing (thin stream only)
                if pace == 0:
                    a_lb = max(self.true_ld, 0.0)
                else:
                    a_lb = max(self.true_ld,
                               min(self.bsde.dim / 5,
                                   min(7, self.bsde.a - 1))
                               - global_step / 40 / self.bsde.dim / pace)
                a_lb = tf.cast(a_lb, tf.float64)

                # ── Logging ──────────────────────────────────────────
                log_freq = self.net_config.logging_frequency
                if step % log_freq == 0 or step == itr - 1:
                    # Validation: use current optimised φ (not sampled)
                    if self.has_actor:
                        phi_val_np = self._masked_phi().numpy()           # [K, J]
                        phi_val_flat = np.tile(
                            phi_val_np.flatten()[np.newaxis, :], (valid_y.shape[0], 1))
                        phi_val_tf = tf.constant(phi_val_flat, dtype=tf.float64)
                    else:
                        phi_val_tf = None
                    v_loss, v_val = self.loss_critic(valid_data,
                                                     training=False,
                                                     use_NN=True,
                                                     phi_flat=phi_val_tf)
                    x0_batch = np.ones([B, self.bsde.dim]) * 2.0
                    cur_val  = self.compute_value(x0_batch).numpy()
                    cur_grad = self.compute_grad(x0_batch).numpy()
                    elapsed  = time.time() - start
                    self.loss_rec.append(float(v_loss))
                    self.val_rec.append(float(v_val))
                    logging.info(
                        'step=%5d  val_loss=%.6f  V(x0)=%.4f  '
                        '‖∇V‖=%.4f  t=%3.0fs',
                        step, float(v_loss), cur_val, cur_grad, elapsed)
                # Solution operator: sample φ ~ Dirichlet for this step.
                # The same trajectories serve as collocation points for all φ;
                # different φ changes B_eff and therefore the HJB coefficients.
                train_data = (
                    a_lb,
                    self.dw_sample[:B, :, cs:ce],
                    self.y_sample[:B, :, csy:cey]
                )
                if self.has_actor:
                    K, J = self.bsde.K, self.bsde.dim
                    mask_np = self.phi_logit_mask.numpy()          # [K, J]: 0 valid, -1e9 invalid
                    phi_sampled = np.zeros((K, J), dtype=np.float64)
                    for k in range(K):
                        valid_j = np.where(mask_np[k] == 0)[0]    # valid classes for pool k
                        alpha_vec = np.ones(len(valid_j)) * self.phi_sample_alpha
                        phi_sampled[k, valid_j] = np.random.dirichlet(alpha_vec)
                    actor_will_update = (
                        global_step >= self.actor_warmup_steps and
                        global_step % self.actor_update_freq == 0)
                    phi_for_critic = (self._masked_phi().numpy()
                                      if actor_will_update else phi_sampled)
                    phi_flat_np = np.tile(
                        phi_for_critic.flatten()[np.newaxis, :], (B, 1))
                    if not hasattr(self, '_phi_flat_var') or \
                            self._phi_flat_var.shape[0] != B:
                        self._phi_flat_var = tf.Variable(
                            phi_flat_np, trainable=False, dtype=tf.float64)
                    else:
                        self._phi_flat_var.assign(phi_flat_np)
                    phi_flat_tf = self._phi_flat_var
                else:
                    phi_flat_tf = None

                self.train_step_critic(train_data, phi_flat=phi_flat_tf)

                if self.has_actor and (global_step % self.actor_update_freq == 0):
                    if global_step < self.actor_warmup_steps:
                        if global_step % (self.actor_update_freq * 10) == 0:
                            logging.info(
                                '  [actor warmup] step=%d < %d',
                                global_step, self.actor_warmup_steps)
                        global_step += 1
                        continue
                    phi_np = self._masked_phi().numpy()  # [K, J]

                    traj, dY, mask = self._gen_actor_traj(B, phi_np)

                    lam = self.adjoint_solver.backward_pass(
                        traj, dY, mask,
                        self.model_critic, phi_np, self.bsde)

                    # Phase 4: actor gradient step
                    a_loss, grads = self.train_step_actor(
                        tf.constant(traj, dtype=tf.float64),
                        tf.constant(lam,  dtype=tf.float64))

                    self.actor_loss_rec.append(float(a_loss))

                    if step % log_freq == 0:
                        phi_now  = self._masked_phi().numpy()
                        phi_init = np.array(self.bsde.phi_star, dtype=np.float64)
                        phi_disp = float(np.linalg.norm(phi_now - phi_init))
                        logging.info('  actor_loss=%.6f  phi_disp=%.4f', float(a_loss), phi_disp)
                global_step += 1

            self.model_critic.NN_value.save_weights(f'models/{suffix}.weights.h5')
            self.model_critic.NN_value_grad.save_weights(f'models/{suffix}_grad.weights.h5')
            if self.has_actor:
                np.save(f'models/{suffix}_phi_logits.npy', self.phi_logits.numpy())
            np.savetxt(
                f'logs_{act_str}/{suffix}',
                np.stack([self.val_rec, self.loss_rec]))

        self.model_critic.NN_value.save_weights(f'models/{suffix}.weights.h5')
        self.model_critic.NN_value_grad.save_weights(f'models/{suffix}_grad.weights.h5')
        if self.has_actor:
            np.save(f'models/{suffix}_phi_logits.npy', self.phi_logits.numpy())
        logging.info('Training complete. Final model saved.')

    # ------------------------------------------------------------------
    # Critic loss and gradient
    # ------------------------------------------------------------------

    def loss_critic(self, inputs, training, use_NN, phi_flat=None):
        delta, cur_val = self.model_critic.forward(
            inputs, training=training, use_NN=use_NN, phi_flat=phi_flat)
        T    = self.eqn_config.total_time_critic
        loss = tf.math.reduce_variance(delta) / (T ** 2)

        # ── G-consistency fix ────────────────────────────────────────────
        # Root cause confirmed by diagnose_jump_kappa(): the BSDE residual
        # only supervises G through the diffusion integral ∫G·dW, which is
        # a 1st-order, direction-averaged signal. Nothing in the original
        # loss constrains G's *local curvature*, so κ_G = ∂G_j/∂x_j|_{x_j=0}
        # can land at a plausible magnitude while pointing nowhere near the
        # true ∂²V/∂x_j²|_{x_j=0} — observed empirically as
        # cosine(κ_G, κ_V) ≈ 0 despite comparable magnitudes.
        #
        # Fix: explicitly pull G toward stop_gradient(autograd(V))
        #   (a) along the full sampled trajectory (broad-coverage anchor), and
        #   (b) at randomly-probed points with one coordinate forced to 0
        #       (directly targets the curvature the jump correction needs).
        # This restores G ≈ ∇V self-consistently, so κ computed from G in
        # adjoint.py's jump correction is trained to agree with the value
        # network's own curvature — matching the self-consistency that
        # Corollary 5.1 / eq. (6.9) requires — rather than switching κ's
        # source to autograd(V), which would abandon that self-consistency
        # (and V's curvature has no special claim to correctness either:
        # both κ_G and κ_V are equally undertrained without this fix).
        #
        # Controlled via eqn_config: consist_weight (default 1.0),
        # consist_bnd_weight (default 1.0), consist_bnd_samples (default 64).
        # Set both weights to 0 to recover the original 803 behaviour exactly.
        consist_weight = float(getattr(self.eqn_config, 'consist_weight', 1.0))
        bnd_weight     = float(getattr(self.eqn_config, 'consist_bnd_weight', 1.0))
        n_bnd          = int(getattr(self.eqn_config, 'consist_bnd_samples', 64))

        if training and use_NN and (consist_weight > 0.0 or bnd_weight > 0.0):
            _, _, y_batch = inputs
            y_batch = tf.cast(y_batch, tf.float64)
            J  = self.bsde.dim
            N1 = tf.shape(y_batch)[2]              # N+1 time points

            # (a) full-trajectory consistency: G vs stop_gradient(autograd(V))
            if consist_weight > 0.0:
                x_traj = tf.reshape(
                    tf.transpose(y_batch, [0, 2, 1]), [-1, J])       # [B*(N+1), J]
                pf_traj = (tf.repeat(phi_flat, N1, axis=0)
                          if phi_flat is not None else None)

                with tf.GradientTape() as tape_c:
                    tape_c.watch(x_traj)
                    v_c = self.model_critic.NN_value(
                        x_traj, phi_flat=pf_traj, training=False)
                ag  = tf.stop_gradient(tape_c.gradient(v_c, x_traj))
                g_c = self.model_critic.NN_value_grad(
                    x_traj, phi_flat=pf_traj, training=training)
                consist_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(g_c - ag), axis=1))
                loss = loss + consist_weight * consist_loss

            # (b) explicit boundary-curvature consistency: κ_G vs stop_gradient(κ_V)
            if bnd_weight > 0.0:
                x_bnd  = tf.random.uniform([n_bnd, J], minval=0.1, maxval=6.0,
                                           dtype=tf.float64)
                j_rand = tf.random.uniform([n_bnd], minval=0, maxval=J,
                                           dtype=tf.int32)
                onehot = tf.one_hot(j_rand, J, dtype=tf.float64)
                x_bnd  = x_bnd * (1.0 - onehot)            # force x_j = 0

                pf_bnd = (tf.repeat(phi_flat[:1], n_bnd, axis=0)
                          if phi_flat is not None else None)

                # κ_V = ∂²V/∂x_j² via double autograd through NN_value (target)
                with tf.GradientTape() as t2:
                    t2.watch(x_bnd)
                    with tf.GradientTape() as t1:
                        t1.watch(x_bnd)
                        v_b = self.model_critic.NN_value(
                            x_bnd, phi_flat=pf_bnd, training=False)
                    dv  = t1.gradient(v_b, x_bnd)
                    dvj = tf.reduce_sum(dv * onehot, axis=1)
                kappa_V_full = tf.stop_gradient(t2.gradient(dvj, x_bnd))
                kappa_V_j    = tf.reduce_sum(kappa_V_full * onehot, axis=1)

                # κ_G = ∂G_j/∂x_j via single autograd through NN_value_grad
                with tf.GradientTape() as t3:
                    t3.watch(x_bnd)
                    g_b    = self.model_critic.NN_value_grad(
                        x_bnd, phi_flat=pf_bnd, training=training)
                    scalar = tf.reduce_sum(g_b * onehot)
                kappa_G_full = t3.gradient(scalar, x_bnd)
                kappa_G_j    = tf.reduce_sum(kappa_G_full * onehot, axis=1)

                bnd_loss = tf.reduce_mean(tf.square(kappa_G_j - kappa_V_j))
                loss = loss + bnd_weight * bnd_loss

        return loss, cur_val

    def grad_critic(self, inputs, training, use_NN, phi_flat=None):
        with tf.GradientTape() as tape:
            loss, _ = self.loss_critic(inputs, training, use_NN,
                                       phi_flat=phi_flat)
        grads = tape.gradient(loss, self.model_critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        return grads

    @tf.function(reduce_retracing=True)
    def train_step_critic(self, train_data, phi_flat=None):
        grads = self.grad_critic(train_data, training=True, use_NN=True,
                                 phi_flat=phi_flat)
        self.optimizer_critic.apply_gradients(
            zip(grads, self.model_critic.trainable_variables))

    # ------------------------------------------------------------------
    # Actor policy gradient and update
    # ------------------------------------------------------------------

    def loss_actor(self, traj_tf, lam_tf, training=True):
        N    = self.eqn_config.num_time_interval_critic
        dt   = self.eqn_config.total_time_critic / N
        # Bug 2 fix: B_mat fixed at fluid-optimal φ* (paper eq. 3.12, Contribution 2)
        B_mat_tf = tf.cast(self.bsde.B_mat,     tf.float64)   # [J, K] fixed
        c_tf     = tf.cast(self.bsde.c_weights, tf.float64)   # [K]
        u_lo     = tf.cast(self.bsde.u_lower,   tf.float64)
        u_hi     = tf.cast(self.bsde.u_upper,   tf.float64)

        phi      = self._masked_phi()  # [K, J], mask applied: invalid positions forced to 0

        # Bug 1 fix: dynamic β(φ̂) = θ − M_cap · (φ − φ*)  (paper eq. 3.12)
        phi_star_tf = tf.cast(self.bsde.phi_star, tf.float64)   # [K, J]
        theta_tf    = tf.cast(self.bsde.theta,    tf.float64)   # [J]
        M_cap_tf    = tf.cast(self.bsde.M_cap,    tf.float64)   # [J, K]
        hat_phi     = phi - phi_star_tf                          # [K, J]
        beta        = theta_tf - tf.reduce_sum(
            M_cap_tf * tf.transpose(hat_phi), axis=1)            # [J]

        K, J = self.bsde.K, self.bsde.dim
        B    = tf.shape(traj_tf)[0]

        # ── Vectorised time loop ───────────────────────────────────────────
        # Single batched NN call with shape [B*N, J] instead of N calls of [B, J].
        #
        # traj_tf: [B, J, N+1]  →  x_all: [B, N, J]  →  x_flat: [B*N, J]
        # lam_tf:  [B, J, N]    →  lam_all: [B, N, J] →  lam_flat: [B*N, J]
        x_all   = tf.transpose(traj_tf[:, :, :N], [0, 2, 1])      # [B, N, J]
        lam_all = tf.transpose(lam_tf[:,  :, :N], [0, 2, 1])      # [B, N, J]
        x_flat   = tf.reshape(x_all,   [-1, J])                   # [B*N, J]
        lam_flat = tf.reshape(lam_all, [-1, J])                   # [B*N, J]

        # phi_flat broadcast: [1, K*J] → [B*N, K*J]
        phi_flat_row = tf.reshape(phi, [1, K * J])
        phi_flat_all = tf.tile(phi_flat_row, [tf.shape(x_flat)[0], 1])  # [B*N, K*J]

        # Single batched critic call
        grad_V_flat = self.model_critic.NN_value_grad(
            x_flat, phi_flat=phi_flat_all, training=False)         # [B*N, J]

        BT_grad_flat = tf.einsum('jk,bj->bk', B_mat_tf, grad_V_flat)  # [B*N, K]
        u_hat_flat   = tf.clip_by_value(BT_grad_flat / (2.0 * c_tf), u_lo, u_hi)

        b_flat = (-tf.einsum('jk,bk->bj', B_mat_tf, u_hat_flat)
                  + beta)                                          # [B*N, J]

        # Per-sample dot and ctrl-cost terms  [B*N]
        dot_flat  = tf.reduce_sum(lam_flat * b_flat,         axis=1)
        ctrl_flat = tf.reduce_sum(c_tf * tf.square(u_hat_flat), axis=1)

        # Reshape to [B, N] for time-weighted mean
        dot_mat  = tf.reshape(dot_flat,  [B, N])              # [B, N]
        ctrl_mat = tf.reshape(ctrl_flat, [B, N])              # [B, N]

        # Precomputed discount factors  [1, N]
        t_idx    = tf.cast(tf.range(N), tf.float64)
        disc_vec = tf.exp(-tf.cast(self.gamma * dt, tf.float64) * t_idx)
        disc_row = tf.reshape(disc_vec, [1, N])

        # total = dt · mean_b( Σ_t disc_t · (dot_bt + ctrl_bt) )
        total = dt * tf.reduce_mean(
            tf.reduce_sum(disc_row * (dot_mat + ctrl_mat), axis=1))

        return total   # minimise → descend on J(φ)

    @tf.function(reduce_retracing=True)
    def train_step_actor(self, traj_np, lam_np):
        with tf.GradientTape() as tape:
            loss = self.loss_actor(traj_np, lam_np, training=True)
        grads = tape.gradient(loss, [self.phi_logits])
        self.optimizer_actor.apply_gradients(zip(grads, [self.phi_logits]))
        # Clip range now configurable (see __init__ / _phi_logit_clip);
        # None means "no clipping at all". Previously hardcoded to
        # [-1.5, 1.5], which was silently determining the converged φ
        # for BOTH jump=True and jump=False (see comment in __init__).
        if self._phi_logit_clip is not None:
            lo, hi = self._phi_logit_clip
            self.phi_logits.assign(tf.clip_by_value(self.phi_logits, lo, hi))
        return loss, grads

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def compute_value(self, x0):
        x_tf = tf.constant(x0, dtype=tf.float64)
        phi_flat = self._current_phi_flat(x0.shape[0]) if self.has_actor else None
        return tf.reduce_mean(
            self.model_critic.NN_value(x_tf, phi_flat=phi_flat, training=False))

    def compute_grad(self, x0):
        x_tf = tf.constant(x0, dtype=tf.float64)
        phi_flat = self._current_phi_flat(x0.shape[0]) if self.has_actor else None
        g = self.model_critic.NN_value_grad(x_tf, phi_flat=phi_flat, training=False)
        return tf.reduce_mean(tf.norm(g, axis=1))
    
    def _masked_phi(self): 
        return tf.nn.softmax(self.phi_logits + self.phi_logit_mask, axis=1)

    def _current_phi_flat(self, B):
        """Return current optimised φ broadcast to [B, K*J] as tf.Tensor."""
        phi_np = self._masked_phi().numpy()
        phi_flat_np = np.tile(phi_np.flatten()[np.newaxis, :], (B, 1))
        return tf.constant(phi_flat_np, dtype=tf.float64)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gen_actor_traj(self, B: int, phi_np: np.ndarray):
        J        = self.bsde.dim
        N        = self.eqn_config.num_time_interval_critic
        T        = self.eqn_config.total_time_critic
        dt       = T / N
        sdt      = np.sqrt(dt)
        refl_tol = float(getattr(self.train_config, 'refl_tol', 1e-8))

        # Current phi's residual drift: β_j(φ̂) = θ_j − Σ_k M_cap[j,k]·φ̂_kj
        hat_phi   = phi_np - self.bsde.phi_star              # [K, J]
        beta_curr = self.bsde.theta - np.einsum(
            'jk,kj->j', self.bsde.M_cap, hat_phi)            # [J]

        # Critic conditioning feature (fixed for all steps — same phi)
        phi_flat_tf = tf.constant(
            np.tile(phi_np.flatten()[np.newaxis, :], (B, 1)),
            dtype=tf.float64)                                 # [B, K*J]

        B_mat_np = np.array(self.bsde.B_mat,     dtype=np.float64)  # [J, K]
        c_np     = np.array(self.bsde.c_weights, dtype=np.float64)  # [K]
        u_lo_np  = np.array(self.bsde.u_lower,   dtype=np.float64)
        u_hi_np  = np.array(self.bsde.u_upper,   dtype=np.float64)

        x0_val = float(getattr(self.eqn_config, 'x0', 0.1))
        x      = np.ones((B, J), dtype=np.float64) * x0_val

        traj = np.zeros((B, J, N + 1), dtype=np.float64)
        dY   = np.zeros((B, J, N),     dtype=np.float64)
        traj[:, :, 0] = x

        for t in range(N):
            # Optimal control at current state
            grad_V = self.model_critic.NN_value_grad(
                tf.constant(x, dtype=tf.float64),
                phi_flat=phi_flat_tf, training=False).numpy()   # [B, J]
            u_hat  = np.clip(
                grad_V @ B_mat_np / (2.0 * c_np),
                u_lo_np, u_hi_np)                               # [B, K]

            # SDE increment under current phi and u*
            drift  = beta_curr[np.newaxis, :] - u_hat @ B_mat_np.T  # [B, J]
            dW     = (self.bsde.Sigma_sqrt @
                      np.random.normal(size=(J, B))).T * sdt         # [B, J]

            # Skorokhod projection
            x_free           = x + drift * dt + dW
            x                = np.maximum(x_free, 0.0)
            dY[:, :, t]      = np.maximum(-x_free, 0.0)
            traj[:, :, t + 1] = x

        mask = (dY > refl_tol).astype(np.float64)
        return traj, dY, mask

    def _build_validation_data(self, B, N, itr):
        """Use the last 20 episodes in the sample buffer as validation."""
        w_list = []
        y_list = []
        for step in range(itr, itr + 20):
            cs,  ce  = step * N,   (step + 1) * N       # dw : N entries
            csy, cey = step * N,   step * N + N + 1     # y  : N+1 entries
            w_list.append(self.dw_sample[:B, :, cs:ce])
            y_list.append(self.y_sample[:B, :, csy:cey])
        return (np.concatenate(w_list, axis=0),
                np.concatenate(y_list, axis=0))

    def _gen_filename(self):
        sim    = getattr(self.eqn_config, 'simulation', '')
        T      = self.eqn_config.total_time_critic
        N      = self.eqn_config.num_time_interval_critic
        itr    = self.net_config.num_iterations
        h0     = self.net_config.num_hiddens_critic[0]
        hL     = self.net_config.num_hiddens_critic[-1]
        act    = self.net_config.activation
        td     = self.train_config.TD_type
        ts     = datetime.now().strftime('%m%d_%H%M')
        suffix = (f"{self.bsde.name}{self.queue_example}"
                  f"_{self.net_config.transformation}"
                  f"_{h0}{hL}_{T}_{N}_{itr}"
                  f"_dim={self.bsde.dim}{sim}_{td}_{act}_{ts}")
        suffix = suffix.replace(' ', '').replace(',', '_')
        suffix = suffix.replace('[', '').replace(']', '')
        return suffix