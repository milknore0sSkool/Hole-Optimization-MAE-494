from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_burgers_mat(path: str | Path):
    """Load classic burgers_shock.mat-style data.

    Returns x (nx,), t (nt,), u (nt,nx).
    """
    data = loadmat(path)
    required = {"x", "t", "usol"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Missing keys in MAT file: {sorted(missing)}")
    x = np.asarray(data["x"]).reshape(-1).astype(np.float64)
    t = np.asarray(data["t"]).reshape(-1).astype(np.float64)
    usol = np.real(np.asarray(data["usol"])).astype(np.float64)
    if usol.shape == (len(x), len(t)):
        u = usol.T
    elif usol.shape == (len(t), len(x)):
        u = usol
    else:
        raise ValueError(f"Unexpected usol shape {usol.shape}; x={len(x)}, t={len(t)}")
    return x, t, u


def load_burgers_npz(path: str | Path):
    data = np.load(path)
    x = np.asarray(data["x"]).reshape(-1).astype(np.float64)
    t = np.asarray(data["t"]).reshape(-1).astype(np.float64)
    usol = np.asarray(data["usol"]).astype(np.float64)
    if usol.shape == (len(x), len(t)):
        u = usol.T
    elif usol.shape == (len(t), len(x)):
        u = usol
    else:
        raise ValueError(f"Unexpected usol shape {usol.shape}")
    return x, t, u
