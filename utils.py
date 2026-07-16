import numpy as np

def build_sigma(sigma_cfg, dim: int) -> np.ndarray:
    """
    Build the covariance matrix Σ from config.
    """
    if isinstance(sigma_cfg, str) and sigma_cfg.lower() == 'identity':
        return np.eye(dim, dtype=np.float64)
    arr = np.array(sigma_cfg, dtype=np.float64)
    assert arr.shape == (dim, dim), \
        f"Sigma must be ({dim},{dim}), got {arr.shape}"
    return arr