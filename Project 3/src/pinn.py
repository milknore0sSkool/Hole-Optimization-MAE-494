from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch


class PINN(torch.nn.Module):
    def __init__(self, width: int = 48, depth: int = 3):
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)


@dataclass
class TrainConfig:
    adam_epochs: int = 500
    n_collocation: int = 700
    n_ic: int = 256
    n_bc: int = 256
    learning_rate: float = 1e-3
    lbfgs_max_iter: int = 120
    seed: int = 0
    width: int = 48
    depth: int = 3
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 1.0
    log_every: int = 25


def pde_residual(model: PINN, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    xt = torch.cat([x, t], dim=1).requires_grad_(True)
    u = model(xt)
    grad = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x = grad[:, 0:1]
    u_t = grad[:, 1:2]
    grad2 = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = grad2[:, 0:1]
    return u_t + u * u_x - nu * u_xx


def _loss_terms(model, nu, tensors):
    x_f, t_f, x0, t0, u0, xb_l, xb_r, tb, ub = tensors
    r = pde_residual(model, x_f, t_f, nu)
    l_pde = torch.mean(r**2)
    l_ic = torch.mean((model(torch.cat([x0, t0], dim=1)) - u0) ** 2)
    l_bc = torch.mean((model(torch.cat([xb_l, tb], dim=1)) - ub) ** 2) + torch.mean(
        (model(torch.cat([xb_r, tb], dim=1)) - ub) ** 2
    )
    return l_pde, l_ic, l_bc


def _make_points(cfg: TrainConfig):
    x_f = -1.0 + 2.0 * torch.rand(cfg.n_collocation, 1)
    t_f = torch.rand(cfg.n_collocation, 1)

    x0 = -1.0 + 2.0 * torch.rand(cfg.n_ic, 1)
    t0 = torch.zeros_like(x0)
    u0 = -torch.sin(math.pi * x0)

    tb = torch.rand(cfg.n_bc, 1)
    xb_l = -torch.ones_like(tb)
    xb_r = torch.ones_like(tb)
    ub = torch.zeros_like(tb)
    return x_f, t_f, x0, t0, u0, xb_l, xb_r, tb, ub


def train_pinn(nu: float, improved: bool, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = PINN(width=cfg.width, depth=cfg.depth)
    tensors = _make_points(cfg)

    history = []
    adam = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()

    for epoch in range(cfg.adam_epochs):
        adam.zero_grad()
        l_pde, l_ic, l_bc = _loss_terms(model, nu, tensors)
        loss = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_bc * l_bc
        loss.backward()
        adam.step()
        if epoch % cfg.log_every == 0 or epoch == cfg.adam_epochs - 1:
            history.append(
                {
                    "stage": "Adam",
                    "step": epoch,
                    "pde": float(l_pde.detach()),
                    "ic": float(l_ic.detach()),
                    "bc": float(l_bc.detach()),
                    "total": float(loss.detach()),
                }
            )

    if improved:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=cfg.lbfgs_max_iter,
            max_eval=cfg.lbfgs_max_iter + 20,
            tolerance_grad=1e-8,
            tolerance_change=1e-10,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        closure_calls = {"n": 0}

        def closure():
            lbfgs.zero_grad()
            l_pde, l_ic, l_bc = _loss_terms(model, nu, tensors)
            loss = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_bc * l_bc
            loss.backward()
            n = closure_calls["n"]
            if n % 5 == 0:
                history.append(
                    {
                        "stage": "L-BFGS",
                        "step": cfg.adam_epochs + n,
                        "pde": float(l_pde.detach()),
                        "ic": float(l_ic.detach()),
                        "bc": float(l_bc.detach()),
                        "total": float(loss.detach()),
                    }
                )
            closure_calls["n"] += 1
            return loss

        lbfgs.step(closure)

    wall = time.perf_counter() - start
    return model, history, wall


def predict_grid(model: PINN, x: np.ndarray, t: np.ndarray):
    X, T = np.meshgrid(x, t, indexing="xy")
    pts = torch.tensor(np.column_stack([X.ravel(), T.ravel()]), dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        pred = model(pts).cpu().numpy().reshape(len(t), len(x))
    wall = time.perf_counter() - start
    return pred, wall


def supervised_capacity_test(
    x: np.ndarray,
    t: np.ndarray,
    u: np.ndarray,
    cfg: TrainConfig,
    n_samples: int = 5000,
    epochs: int = 1500,
):
    """Fit the same network directly to reference solution values."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = PINN(width=cfg.width, depth=cfg.depth)

    ix = rng.integers(0, len(x), n_samples)
    it = rng.integers(0, len(t), n_samples)
    X_train = torch.tensor(np.column_stack([x[ix], t[it]]), dtype=torch.float32)
    y_train = torch.tensor(u[it, ix, None], dtype=torch.float32)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    start = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.mean((model(X_train) - y_train) ** 2)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - start
    pred, inference = predict_grid(model, x, t)
    rel = float(np.linalg.norm(pred - u) / np.linalg.norm(u))
    return model, rel, wall, inference
