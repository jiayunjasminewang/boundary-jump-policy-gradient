import json, os
import h5py
import numpy as np
import tensorflow as tf
from scipy import stats

# Paths (edit before running)
CONFIG      = "configs/your_100d_config.json"
BASE        = "models/your_100d_run_prefix"
STEM_JUMP   = BASE + "_run_a"
STEM_NOJUMP = BASE + "_run_b"

N_MC    = 8_000
T_MC    = 60.0
N_STEPS = 3_200
SEED    = 0

# 100D trajectories are simulated in batches to bound memory use.
BATCH_SIZE = int(os.environ.get("EVAL100_BATCH_SIZE", "256"))

with open(CONFIG) as f:
    cfg = json.load(f)

eq  = cfg["eqn_config"]
net = cfg["net_config"]

DIM      = int(eq["dim"])
K        = int(eq["K"])
HIDDENS  = net["num_hiddens_critic"]
ACT      = net["activation"]
PHI_ENC  = int(net["phi_enc_hidden"])

def _parse_matrix_field(val, shape, name):
    # Some configs use "identity" instead of an explicit matrix (e.g. Sigma).
    if isinstance(val, str):
        s = val.strip().lower()
        if s == "identity":
            if shape is None or len(shape) != 2 or shape[0] != shape[1]:
                raise ValueError(f"'{name}': 'identity' shorthand requires a "
                                  f"square shape, got {shape}")
            return np.eye(shape[0], dtype=np.float64)
        raise ValueError(f"'{name}': unrecognized string spec {val!r} "
                          "(expected a matrix/array or the string 'identity')")
    return np.array(val, dtype=np.float64)

def _cfg_array(key, shape=None):
    if key not in eq:
        raise KeyError(f"eq_config is missing required field '{key}'")
    val = eq[key]
    try:
        return _parse_matrix_field(val, shape, key)
    except Exception as ex:
        raise ValueError(f"Could not parse eq_config['{key}'] = {val!r}  ({ex})")

M_cap    = _cfg_array("M_cap")
phi_star = _cfg_array("phi_star")
phi_mask = np.array(eq.get("phi_mask", (M_cap > 0).astype(float)), dtype=np.float64)
beta_cfg = _cfg_array("beta")
Sigma    = _cfg_array("Sigma", shape=(DIM, DIM))
w        = _cfg_array("w")
c        = _cfg_array("c")
u_lower  = _cfg_array("u_lower")
u_upper  = _cfg_array("u_upper")
gamma    = float(eq["discount"])
x0_val   = float(eq.get("x0", 1.0))

u_star   = _cfg_array("u_star") if "u_star" in eq else np.ones(K, dtype=np.float64)

Sigma_chol = np.linalg.cholesky(Sigma)


def B_of_phi(phi):
    return (M_cap * phi).T

B_FIXED = B_of_phi(phi_star)


def find_file(name):
    for base in [".", "models"]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"'{name}' not found")


def _read_dense_layers(h5_group):
    keys = sorted(h5_group.keys(),
                  key=lambda k: int(k.split('_')[1]) if '_' in k else 0)
    result = []
    for k in keys:
        var_grp = h5_group[k]['vars']
        var_keys = sorted(var_grp.keys(), key=int)
        result.append([np.array(var_grp[vk]) for vk in var_keys])
    return result

def _assert_weight_sets_match(layers, weight_sets, path, label):
    if len(weight_sets) != len(layers):
        raise ValueError(
            f"{label} weight count mismatch in {path}: "
            f"file has {len(weight_sets)} layers, model expects {len(layers)}.")
    for idx, (layer, weights) in enumerate(zip(layers, weight_sets)):
        expected = [tuple(w.shape) for w in layer.get_weights()]
        actual   = [tuple(w.shape) for w in weights]
        if actual != expected:
            raise ValueError(
                f"{label} layer {idx} shape mismatch in {path}: "
                f"file has {actual}, model expects {expected}.")
        for w_idx, arr in enumerate(weights):
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{label} layer {idx} tensor {w_idx} in {path} contains NaN or Inf.")

def _set_weights_strict(layers, weight_sets, path, label):
    _assert_weight_sets_match(layers, weight_sets, path, label)
    for layer, weights in zip(layers, weight_sets):
        layer.set_weights(weights)

def load_value_weights(model, path):
    with h5py.File(path, 'r') as f:
        if 'dense_layers' not in f:
            raise KeyError(f"Cannot locate dense_layers in {path}. Keys: {list(f.keys())}")
        layer_weights = _read_dense_layers(f['dense_layers'])
    _set_weights_strict(model.dense_layers, layer_weights, path, "value")

def _inspect_h5(path):
    def _tree(grp, indent=0, depth=0):
        if depth > 5:
            return
        for k in sorted(grp.keys()):
            v = grp[k]
            if hasattr(v, 'shape'):
                print(' ' * indent + f'{k}: {v.shape}')
            else:
                print(' ' * indent + f'{k}/')
                _tree(v, indent + 2, depth + 1)
    with h5py.File(path, 'r') as f:
        print(f'\n=== {path} ===')
        _tree(f)

def _read_phi_encoder(f, path):
    if 'phi_encoder' in f:
        return _read_dense_layers(f['phi_encoder'])
    try:
        return _read_dense_layers(f['layers']['sequential']['layers'])
    except KeyError:
        pass
    try:
        seq_grp = f['layers']['sequential']
        deps = seq_grp['_layer_checkpoint_dependencies']
        return _read_dense_layers(deps)
    except KeyError:
        _inspect_h5(path)
        raise KeyError(f"Cannot locate phi_encoder in {path}. Top-level keys: {list(f.keys())}")

def load_grad_weights(model, path):
    with h5py.File(path, 'r') as f:
        if 'dense_layers' not in f:
            raise KeyError(f"Cannot locate dense_layers in {path}. Keys: {list(f.keys())}")
        main_weights = _read_dense_layers(f['dense_layers'])
        enc_weights = _read_phi_encoder(f, path)
    _set_weights_strict(model.dense_layers, main_weights, path, "grad/main")
    _set_weights_strict(model.phi_encoder.layers, enc_weights, path, "grad/phi_encoder")


class ValueNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layers = (
            [tf.keras.layers.Dense(h, use_bias=False, dtype=tf.float64)
             for h in HIDDENS]
            + [tf.keras.layers.Dense(1, use_bias=True, dtype=tf.float64)]
        )
    def call(self, x, phi_flat, training=False):
        x  = tf.cast(x,        tf.float64)
        pf = tf.cast(phi_flat, tf.float64)
        y  = tf.concat([tf.square(x), pf], axis=1)
        for layer in self.dense_layers[:-1]:
            y = tf.nn.elu(layer(y)) if ACT == "elu" else tf.nn.relu(layer(y))
        return self.dense_layers[-1](y)

class GradNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.phi_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(PHI_ENC, use_bias=True, dtype=tf.float64,
                                  activation='elu' if ACT == 'elu' else 'relu'),
            tf.keras.layers.Dense(PHI_ENC, use_bias=True, dtype=tf.float64,
                                  activation='elu' if ACT == 'elu' else 'relu'),
        ])
        self.dense_layers = (
            [tf.keras.layers.Dense(h, use_bias=False, dtype=tf.float64)
             for h in HIDDENS]
            + [tf.keras.layers.Dense(DIM, use_bias=False, dtype=tf.float64)]
        )
    def call(self, x, phi_flat=None, training=False):
        x   = tf.cast(x, tf.float64)
        z   = tf.square(x)
        if phi_flat is not None:
            enc = self.phi_encoder(tf.cast(phi_flat, tf.float64))
            y   = tf.concat([z, enc], axis=1)
        else:
            y = z
        for layer in self.dense_layers[:-1]:
            y = tf.nn.elu(layer(y)) if ACT == "elu" else tf.nn.relu(layer(y))
        return 2.0 * x * self.dense_layers[-1](y)

def _dummy():
    return (tf.zeros([1, DIM],   dtype=tf.float64),
            tf.zeros([1, K*DIM], dtype=tf.float64))

def build_value(path):
    m = ValueNet(); m(*_dummy()); load_value_weights(m, path); return m

def build_grad(path):
    m = GradNet();  m(*_dummy()); load_grad_weights(m, path);  return m


def logits_to_phi(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (K, DIM):
        raise ValueError(
            f"phi logits shape mismatch: got {logits.shape}, expected {(K, DIM)}.")
    if not np.all(np.isfinite(logits)):
        raise ValueError("phi logits contain NaN or Inf.")
    phi = np.zeros_like(logits)
    for k in range(K):
        valid = np.where(phi_mask[k] > 0)[0]
        if len(valid) == 0:
            raise ValueError(f"phi_mask row {k} has no valid routing targets.")
        row   = logits[k, valid] - logits[k, valid].max()
        ex    = np.exp(row)
        phi[k, valid] = ex / ex.sum()
    row_sums = phi.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError(f"decoded phi rows do not sum to 1: {row_sums}")
    if np.any((phi_mask <= 0) & (np.abs(phi) > 1e-12)):
        raise ValueError("decoded phi has positive mass outside phi_mask.")
    return phi


def eval_value_samples(value_net, phi, n_sample=2048):
    rng_eval = np.random.default_rng(99)
    x_fixed  = np.full((n_sample // 2, DIM), x0_val, dtype=np.float64)
    x_random = rng_eval.uniform(x0_val, 4.0 * x0_val,
                                size=(n_sample - n_sample // 2, DIM))
    x  = np.vstack([x_fixed, x_random]).astype(np.float64)
    pf = np.tile(phi.flatten()[None, :], (n_sample, 1))
    return value_net(tf.constant(x, dtype=tf.float64),
                      phi_flat=tf.constant(pf, dtype=tf.float64),
                      training=False).numpy().ravel()

def cross_eval_table(value_j, phi_j, value_n, phi_n, n_sample=2048):
    results = {}
    for cname, vnet in [("jump", value_j), ("nojump", value_n)]:
        for pname, phi in [("jump", phi_j), ("nojump", phi_n)]:
            results[(cname, pname)] = eval_value_samples(vnet, phi, n_sample).mean()
    return results


def _beta_eff(phi):
    phi_hat = phi - phi_star
    return beta_cfg - (u_star[:, None] * M_cap * phi_hat).sum(axis=0)


def mc_cost_uzero(phi, n_paths=N_MC, batch_size=BATCH_SIZE):
    rng = np.random.default_rng(SEED)
    dt  = T_MC / N_STEPS
    sdt = np.sqrt(dt)
    beta_eff = _beta_eff(phi)

    costs = []
    done = 0
    while done < n_paths:
        n = min(batch_size, n_paths - done)
        x    = np.full((n, DIM), x0_val, dtype=np.float64)
        cost = np.zeros(n, dtype=np.float64)
        disc = 1.0
        for _ in range(N_STEPS):
            if disc < 1e-7:
                break
            cost += disc * (x @ w) * dt
            dW    = rng.standard_normal((n, DIM)) @ Sigma_chol.T
            x     = np.maximum(x + beta_eff[None, :] * dt + dW * sdt, 0.0)
            disc *= np.exp(-gamma * dt)
        costs.append(cost)
        done += n
    return np.concatenate(costs)


def mc_cost_controlled(phi, grad_net, value_net, n_paths=N_MC, batch_size=BATCH_SIZE):
    rng = np.random.default_rng(SEED)
    dt  = T_MC / N_STEPS
    sdt = np.sqrt(dt)
    beta_eff = _beta_eff(phi)
    disc_T   = np.exp(-gamma * T_MC)

    costs = []
    done = 0
    while done < n_paths:
        n = min(batch_size, n_paths - done)
        pf_np = np.tile(phi.flatten()[None, :], (n, 1))
        pf_tf = tf.constant(pf_np, dtype=tf.float64)
        x    = np.full((n, DIM), x0_val, dtype=np.float64)
        cost = np.zeros(n, dtype=np.float64)
        disc = 1.0
        for _ in range(N_STEPS):
            if disc < 1e-7:
                break
            grad_V = grad_net(tf.constant(x, dtype=tf.float64), phi_flat=pf_tf).numpy()
            u_hat  = np.clip(grad_V @ B_FIXED / (2.0 * c), u_lower, u_upper)
            cost  += disc * (x @ w + (u_hat ** 2) @ c) * dt

            dW    = rng.standard_normal((n, DIM)) @ Sigma_chol.T
            drift = beta_eff[None, :] - u_hat @ B_FIXED.T
            x     = np.maximum(x + drift * dt + dW * sdt, 0.0)
            disc *= np.exp(-gamma * dt)

        tail = value_net(tf.constant(x, dtype=tf.float64), phi_flat=pf_tf,
                         training=False).numpy().ravel()
        cost += disc_T * tail
        costs.append(cost)
        done += n
    return np.concatenate(costs)


def _mean_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def main():
    np.random.seed(SEED); tf.random.set_seed(SEED)
    phi_j = logits_to_phi(np.load(find_file(STEM_JUMP   + "_phi_logits.npy")))
    phi_n = logits_to_phi(np.load(find_file(STEM_NOJUMP + "_phi_logits.npy")))
    print(f"phi (jump):\n{np.round(phi_j, 4)}")
    print(f"phi (nojump):\n{np.round(phi_n, 4)}")

    value_j = build_value(find_file(STEM_JUMP   + ".weights.h5"))
    grad_j  = build_grad( find_file(STEM_JUMP   + "_grad.weights.h5"))
    value_n = build_value(find_file(STEM_NOJUMP + ".weights.h5"))
    grad_n  = build_grad( find_file(STEM_NOJUMP + "_grad.weights.h5"))

    tbl = cross_eval_table(value_j, phi_j, value_n, phi_n)
    Vjj, Vjn = tbl[("jump","jump")],   tbl[("jump","nojump")]
    Vnj, Vnn = tbl[("nojump","jump")], tbl[("nojump","nojump")]
    agree = (Vjj < Vjn) and (Vnn < Vnj)

    cells = {
        "jump":   {"jump": None, "nojump": None},
        "nojump": {"jump": None, "nojump": None},
    }
    for crit_label, gnet, vnet in [("jump", grad_j, value_j), ("nojump", grad_n, value_n)]:
        for phi_label, phi in [("jump", phi_j), ("nojump", phi_n)]:
            cells[crit_label][phi_label] = mc_cost_controlled(phi, gnet, vnet)

    phi_wins = {"jump": 0, "nojump": 0}
    for crit_label in ["jump", "nojump"]:
        cj, cn = cells[crit_label]["jump"], cells[crit_label]["nojump"]
        diff = cj - cn
        md   = diff.mean()
        t, p = stats.ttest_1samp(diff, 0)
        sig  = p < 0.05
        winner = "jump" if (sig and md < 0) else ("nojump" if (sig and md > 0) else "tie")
        if sig and md < 0:  phi_wins["jump"]   += 1
        elif sig and md > 0: phi_wins["nojump"] += 1
        print(f"critic={crit_label:7s}  jump={cj.mean():.4f}+-{cj.std()/np.sqrt(len(cj)):.4f}  "
              f"nojump={cn.mean():.4f}+-{cn.std()/np.sqrt(len(cn)):.4f}  "
              f"diff={md:+.4f} p={p:.3f}  winner={winner}")

    cj_diag, cn_diag = cells["jump"]["jump"], cells["nojump"]["nojump"]
    diff_diag = cj_diag - cn_diag
    md_d, se_d = _mean_se(diff_diag)
    t_d, p_d = stats.ttest_1samp(diff_diag, 0)
    ci_lo, ci_hi = md_d - 1.96*se_d, md_d + 1.96*se_d
    print(f"diagonal  jump={cj_diag.mean():.4f}+-{cj_diag.std()/np.sqrt(len(cj_diag)):.4f}  "
          f"nojump={cn_diag.mean():.4f}+-{cn_diag.std()/np.sqrt(len(cn_diag)):.4f}  "
          f"diff={md_d:+.4f} CI=[{ci_lo:+.4f},{ci_hi:+.4f}] p={p_d:.4f}")

    cost_j0 = mc_cost_uzero(phi_j)
    cost_n0 = mc_cost_uzero(phi_n)
    diff_0  = cost_j0 - cost_n0
    md_0, se_0 = _mean_se(diff_0)
    t_0, p_0 = stats.ttest_1samp(diff_0, 0)
    ci0_lo, ci0_hi = md_0 - 1.96*se_0, md_0 + 1.96*se_0
    print(f"u=0       jump={cost_j0.mean():.4f}+-{cost_j0.std()/np.sqrt(len(cost_j0)):.4f}  "
          f"nojump={cost_n0.mean():.4f}+-{cost_n0.std()/np.sqrt(len(cost_n0)):.4f}  "
          f"diff={md_0:+.4f} CI=[{ci0_lo:+.4f},{ci0_hi:+.4f}] p={p_0:.4f}")

    print(f"agree={agree}  phi_wins={phi_wins}  diagonal_diff={md_d:+.4f} p={p_d:.4f}")


if __name__ == "__main__":
    main()
