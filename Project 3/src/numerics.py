from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.linalg import solve_banded


@dataclass
class FVSolution:
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray  # shape (nt, nx)
    wall_seconds: float


def _convective_rhs(u: np.ndarray, dx: float) -> np.ndarray:
    """Rusanov finite-volume discretization of -d(u^2/2)/dx."""
    f = 0.5 * u * u
    a = np.maximum(np.abs(u[:-1]), np.abs(u[1:]))
    flux = 0.5 * (f[:-1] + f[1:]) - 0.5 * a * (u[1:] - u[:-1])
    rhs = np.zeros_like(u)
    rhs[1:-1] = -(flux[1:] - flux[:-1]) / dx
    return rhs


def _implicit_diffusion_step(rhs_full: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Solve (I - dt*nu*Dxx)u_new = rhs with homogeneous Dirichlet boundaries."""
    n = len(rhs_full) - 2
    r = dt * nu / (dx * dx)
    ab = np.zeros((3, n), dtype=np.float64)
    ab[0, 1:] = -r
    ab[1, :] = 1.0 + 2.0 * r
    ab[2, :-1] = -r
    interior = solve_banded((1, 1), ab, rhs_full[1:-1], overwrite_ab=True, check_finite=False)
    out = np.zeros_like(rhs_full)
    out[1:-1] = interior
    return out


def solve_burgers_fv(
    nu: float,
    nx: int = 1025,
    nt: int = 101,
    tmax: float = 1.0,
    cfl: float = 0.25,
) -> FVSolution:
    """Semi-implicit finite-volume solver for viscous Burgers equation.

    PDE: u_t + u u_x - nu u_xx = 0
    Domain: x in [-1,1], t in [0,tmax]
    IC: u(x,0) = -sin(pi x)
    BC: u(-1,t)=u(1,t)=0

    Convection is explicit Rusanov; diffusion is backward-Euler implicit.
    """
    if nx < 5 or nt < 2:
        raise ValueError("nx >= 5 and nt >= 2 are required")

    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    t_save = np.linspace(0.0, tmax, nt, dtype=np.float64)
    dx = float(x[1] - x[0])

    u = -np.sin(np.pi * x)
    u[[0, -1]] = 0.0
    out = np.empty((nt, nx), dtype=np.float64)
    out[0] = u

    time_now = 0.0
    save_idx = 1
    start = time.perf_counter()
    while save_idx < nt:
        target = float(t_save[save_idx])
        while time_now < target - 1e-14:
            umax = max(float(np.max(np.abs(u))), 1e-10)
            dt = min(cfl * dx / umax, target - time_now)
            rhs = u + dt * _convective_rhs(u, dx)
            rhs[[0, -1]] = 0.0
            u = _implicit_diffusion_step(rhs, dt, dx, nu)
            time_now += dt
        out[save_idx] = u
        save_idx += 1

    return FVSolution(x=x, t=t_save, u=out, wall_seconds=time.perf_counter() - start)


def interpolate_reference_to_x(reference: FVSolution, x_target: np.ndarray) -> np.ndarray:
    return np.vstack([np.interp(x_target, reference.x, row) for row in reference.u])


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def save_reference_files(sol: FVSolution, out_dir: str | Path, tag: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"burgers_reference_{tag}.npz"
    mat_path = out_dir / f"burgers_reference_{tag}.mat"
    np.savez_compressed(npz_path, x=sol.x, t=sol.t, usol=sol.u.T)
    savemat(mat_path, {"x": sol.x[:, None], "t": sol.t[:, None], "usol": sol.u.T})
    return npz_path, mat_path
