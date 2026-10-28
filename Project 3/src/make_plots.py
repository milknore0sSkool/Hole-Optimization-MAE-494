from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def finish(ax):
    ax.grid(True, which="both", alpha=0.25)
    ax.set_axisbelow(True)


def main():
    metrics = pd.read_csv(RESULTS / "metrics.csv")
    hist = pd.read_csv(RESULTS / "loss_history.csv")

    viscosities = [0.05, 0.01, 0.01 / np.pi]
    viscosity_labels = ["0.05", "0.01", r"$0.01/\pi$"]
    positions = np.arange(len(viscosities))

    # P3: error versus difficulty parameter.
    methods = [
        ("finite_volume_257", "Finite volume", "o", "-"),
        ("PINN_Adam", "PINN - Adam", "s", "--"),
        ("PINN_Adam_LBFGS", "PINN - Adam + L-BFGS", "^", "-"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, marker, linestyle in methods:
        values = []
        for nu in viscosities:
            row = metrics[(metrics.method == method) & np.isclose(metrics.nu, nu)]
            values.append(float(row.relative_l2.iloc[0]))
        ax.semilogy(
            positions,
            values,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=7,
            label=label,
        )
    ax.set_xticks(positions, viscosity_labels)
    ax.set_xlabel("Viscosity, $\\nu$ (easy to hard)")
    ax.set_ylabel("Relative $L_2$ error")
    ax.set_title("Relative Error Versus Viscosity")
    ax.legend()
    finish(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "p3_error_vs_viscosity.png", dpi=200)
    plt.close(fig)

    # P2: loss components for the hard case.
    hard_tag = "0.01_over_pi"
    sub = hist[(hist.nu_tag == hard_tag) & (hist.method == "PINN_Adam_LBFGS")].sort_values("step")
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [
        ("pde", "PDE residual"),
        ("ic", "Initial condition"),
        ("bc", "Boundary condition"),
    ]:
        ax.semilogy(sub.step, sub[key], linewidth=2, label=label)
    ax.set_xlabel("Optimization step / function evaluation")
    ax.set_ylabel("Mean-squared loss component")
    ax.set_title("Loss Components for the Hard Case")
    ax.axvline(500, linestyle=":", linewidth=1.5, color="0.4")
    ax.text(505, ax.get_ylim()[1] / 1.4, "L-BFGS refinement", fontsize=9, color="0.35", va="top")
    ax.legend()
    finish(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_loss_components_hard.png", dpi=200)
    plt.close(fig)

    # P4: hard-case pointwise error fields.
    ref = np.load(RESULTS / f"reference_eval_{hard_tag}.npz")
    pred_adam = np.load(RESULTS / f"prediction_PINN_Adam_{hard_tag}.npz")["u"]
    pred_lbfgs = np.load(RESULTS / f"prediction_PINN_Adam_LBFGS_{hard_tag}.npz")["u"]
    x_grid, t_grid, u_ref = ref["x"], ref["t"], ref["u"]
    vmax = max(np.abs(pred_adam - u_ref).max(), np.abs(pred_lbfgs - u_ref).max())

    for pred, filename, title in [
        (pred_adam, "p4_error_field_adam.png", "Pointwise Absolute Error - Adam"),
        (pred_lbfgs, "p4_error_field_adam_lbfgs.png", "Pointwise Absolute Error - Adam + L-BFGS"),
    ]:
        error = np.abs(pred - u_ref)
        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(
            error,
            origin="lower",
            aspect="auto",
            extent=[x_grid.min(), x_grid.max(), t_grid.min(), t_grid.max()],
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Absolute error")
        fig.tight_layout()
        fig.savefig(FIGURES / filename, dpi=200)
        plt.close(fig)

    # P4: final-time Fourier error spectrum.
    fig, ax = plt.subplots(figsize=(8, 5))
    for pred, label, linestyle in [
        (pred_adam, "Adam", "--"),
        (pred_lbfgs, "Adam + L-BFGS", "-"),
    ]:
        error = pred[-1] - u_ref[-1]
        spectrum = np.abs(np.fft.rfft(error))
        frequency = np.fft.rfftfreq(len(error), d=(x_grid[1] - x_grid[0]))
        ax.semilogy(
            frequency[1:],
            spectrum[1:] + 1e-14,
            linewidth=2,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel(r"$|\mathrm{FFT}(e)|$")
    ax.set_title("Final-Time Error Spectrum for the Hard Case")
    ax.legend()
    finish(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "p4_error_spectrum_hard.png", dpi=200)
    plt.close(fig)

    # P5: PINN training wall time.
    rows = []
    for nu in viscosities:
        for method in ["PINN_Adam", "PINN_Adam_LBFGS"]:
            rows.append(metrics[(metrics.method == method) & np.isclose(metrics.nu, nu)].iloc[0])
    ordered = pd.DataFrame(rows)
    bar_labels = []
    for _, row in ordered.iterrows():
        nu_label = "0.01/pi" if np.isclose(row.nu, 0.01 / np.pi) else f"{row.nu:g}"
        method_label = "Adam" if row.method == "PINN_Adam" else "Adam + L-BFGS"
        bar_labels.append(f"{nu_label}\n{method_label}")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(np.arange(len(ordered)), ordered.train_seconds)
    ax.set_xticks(np.arange(len(ordered)), bar_labels)
    ax.set_ylabel("Training wall time (s)")
    ax.set_title("PINN Training Cost")
    ax.bar_label(bars, fmt="%.2f s", padding=3, fontsize=9)
    finish(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "p5_training_cost.png", dpi=200)
    plt.close(fig)

    # Hard-case final-time profile.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_grid, u_ref[-1], linewidth=2.5, label="Finite-volume reference")
    ax.plot(x_grid, pred_adam[-1], linestyle="--", linewidth=2, label="PINN - Adam")
    ax.plot(x_grid, pred_lbfgs[-1], linewidth=2, label="PINN - Adam + L-BFGS")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$u(x,t=1)$")
    ax.set_title("Hard-Case Final-Time Solution")
    ax.legend()
    finish(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "hard_case_final_time.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
