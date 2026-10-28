from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import load_burgers_mat
from numerics import interpolate_reference_to_x, relative_l2, save_reference_files, solve_burgers_fv
from pinn import TrainConfig, predict_grid, supervised_capacity_test, train_pinn


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


def tag_for_nu(nu: float) -> str:
    if np.isclose(nu, 0.01 / np.pi):
        return "0.01_over_pi"
    return f"{nu:g}".replace(".", "p")


def main():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    viscosities = [0.05, 0.01, 0.01 / math.pi]
    cfg = TrainConfig()
    metrics = []
    all_hist = []

    for nu in viscosities:
        tag = tag_for_nu(nu)
        print(f"\n=== nu={nu:.8g} ({tag}) ===")

        # High-resolution reference used for quantitative error.
        ref = solve_burgers_fv(nu, nx=1025, nt=101)
        save_reference_files(ref, DATA, tag)

        # Conventional baseline at quarter the spatial resolution.
        conventional = solve_burgers_fv(nu, nx=257, nt=101)
        ref_on_conv = interpolate_reference_to_x(ref, conventional.x)
        conventional_error = relative_l2(conventional.u, ref_on_conv)
        print(f"FV baseline: relL2={conventional_error:.4e}, time={conventional.wall_seconds:.3f}s")

        # Use 257 points for PINN evaluation and stored predictions.
        x_eval = conventional.x
        t_eval = conventional.t
        ref_eval = ref_on_conv
        np.savez_compressed(RESULTS / f"reference_eval_{tag}.npz", x=x_eval, t=t_eval, u=ref_eval)

        metrics.append({
            "nu": nu,
            "nu_tag": tag,
            "method": "finite_volume_257",
            "relative_l2": conventional_error,
            "train_seconds": 0.0,
            "inference_seconds": conventional.wall_seconds,
        })

        for improved, label in [(False, "PINN_Adam"), (True, "PINN_Adam_LBFGS")]:
            model, hist, train_seconds = train_pinn(nu, improved=improved, cfg=cfg)
            pred, inference_seconds = predict_grid(model, x_eval, t_eval)
            error = relative_l2(pred, ref_eval)
            print(f"{label}: relL2={error:.4f}, train={train_seconds:.2f}s, infer={inference_seconds:.4f}s")

            torch.save(model.state_dict(), MODELS / f"{label}_{tag}.pt")
            np.savez_compressed(RESULTS / f"prediction_{label}_{tag}.npz", x=x_eval, t=t_eval, u=pred)
            for row in hist:
                all_hist.append({"nu": nu, "nu_tag": tag, "method": label, **row})
            metrics.append({
                "nu": nu,
                "nu_tag": tag,
                "method": label,
                "relative_l2": error,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
            })

    # Expressivity-vs-optimization diagnostic at hardest viscosity.
    hard_nu = 0.01 / math.pi
    hard_tag = tag_for_nu(hard_nu)
    hard = np.load(RESULTS / f"reference_eval_{hard_tag}.npz")
    cap_model, cap_error, cap_train, cap_inf = supervised_capacity_test(
        hard["x"], hard["t"], hard["u"], cfg
    )
    torch.save(cap_model.state_dict(), MODELS / f"supervised_capacity_{hard_tag}.pt")
    metrics.append({
        "nu": hard_nu,
        "nu_tag": hard_tag,
        "method": "supervised_capacity_test",
        "relative_l2": cap_error,
        "train_seconds": cap_train,
        "inference_seconds": cap_inf,
    })
    print(f"Capacity test: relL2={cap_error:.4f}, train={cap_train:.2f}s")

    # Optional external public-data validation for the canonical hard case.
    public_path = DATA / "burgers_shock.mat"
    validation = {"public_dataset_found": public_path.exists()}
    if public_path.exists():
        x_pub, t_pub, u_pub = load_burgers_mat(public_path)
        hard_ref = solve_burgers_fv(hard_nu, nx=1025, nt=len(t_pub), tmax=float(t_pub[-1]))
        our_on_public = interpolate_reference_to_x(hard_ref, x_pub)
        validation["fv_vs_public_relative_l2"] = relative_l2(our_on_public, u_pub)
        validation["public_shape"] = list(u_pub.shape)
        print(f"External dataset validation: relL2={validation['fv_vs_public_relative_l2']:.4e}")

    pd.DataFrame(metrics).to_csv(RESULTS / "metrics.csv", index=False)
    pd.DataFrame(all_hist).to_csv(RESULTS / "loss_history.csv", index=False)
    (RESULTS / "external_validation.json").write_text(json.dumps(validation, indent=2))
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "CPU",
        "processor": platform.processor(),
        "threads_used": torch.get_num_threads(),
        "seed": cfg.seed,
        "config": cfg.__dict__,
    }
    (RESULTS / "environment.json").write_text(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
