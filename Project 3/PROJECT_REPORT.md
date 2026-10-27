# Optimizing PINN Training for the 1-D Viscous Burgers Equation

**Path:** C — diagnose a failure and demonstrate a targeted fix  
**Method under test:** Physics-informed neural network (PINN)  
**Targeted fix:** two-stage Adam → L-BFGS optimization  
**Difficulty knob:** viscosity \(\nu\)

---

## Executive summary

This case study asks whether PINN failure on a convection-dominated nonlinear PDE can be attributed primarily to insufficient network capacity or to the optimization induced by the physics-informed loss. The minimal problem is the one-dimensional viscous Burgers equation. Viscosity is decreased from \(0.05\) to \(0.01/\pi\), creating a progressively sharper internal layer.

A conventional finite-volume solver remains accurate across the tested range. In contrast, the Adam-only PINN deteriorates from **13.9%** relative L2 error at \(\nu=0.05\) to **42.5%** at \(\nu=0.01/\pi\). Keeping the neural architecture, training points, random seed, and fixed loss weights unchanged but following Adam with L-BFGS reduces error to **1.33%**, **15.8%**, and **28.6%** for the three viscosity values. The relative error reduction due to the optimizer change is approximately **90%**, **59%**, and **33%**, respectively.

A controlled supervised-capacity test on the hardest case fits the same network directly to reference solution values and reaches **5.57%** relative L2 error. Because the architecture can represent the solution far better than the physics-informed training achieves, lack of expressivity alone does not explain the PINN failure. The evidence supports an optimization-dominated diagnosis, although the remaining error at low viscosity indicates that sharp-gradient/spectral effects also become important.

**Recommendation:** do not use a basic Adam-only PINN for this type of sharp Burgers solution. If a PINN is needed for an inverse problem or sparse-data setting, use a second-order/quasi-Newton refinement stage and validate against a conventional solver. For a single forward solve, the conventional method is still the better tool.

---

# 1. Outreach log

This section must be completed from the team's real outreach. No interview is fabricated in this repository.

See `OUTREACH_AND_INTERVIEW.md` for the exact fields to fill in.

---

# 2. Problem brief

## Scientific problem

The viscous Burgers equation is a minimal nonlinear transport-diffusion model:

\[
 u_t + u u_x - \nu u_{xx}=0,
\]

where \(u(x,t)\) is the transported field and \(\nu\) controls diffusion. As \(\nu\) decreases, nonlinear steepening dominates diffusion and a thin internal layer forms. This creates the same type of sharp-gradient difficulty that appears in convection-dominated transport problems.

The canonical public PINN reference problem uses

\[
 x\in[-1,1],\quad t\in[0,1],\quad
 u(x,0)=-\sin(\pi x),\quad
 u(-1,t)=u(1,t)=0,
\]

with the hard-case viscosity \(\nu=0.01/\pi\).

## Computational pain point

A basic PINN is attractive because it does not require a mesh-based residual assembly and can blend data and physics in a single optimization problem. However, the nonlinear PDE residual introduces a difficult loss landscape. The key engineering question is therefore not merely whether a network can represent \(u\), but whether the physics-informed optimization can reliably find that representation as the solution becomes sharper.

## Public data

The external validation dataset is `burgers_shock.mat`, approximately 203 KB, distributed with the original PINN Burgers example. It contains the spatial grid `x`, time grid `t`, and solution field `usol`. The repository includes `src/download_data.py` and a MATLAB loader.

For reproducibility even without network access, the project also independently generates conventional reference solutions in both `.npz` and `.mat` formats.

---

# 3. Minimal case formulation

## Governing equation

\[
\boxed{u_t + u u_x - \nu u_{xx}=0}
\]

with

\[
 u(x,0)=-\sin(\pi x),\qquad u(-1,t)=u(1,t)=0.
\]

## Difficulty knob

Viscosity is the controlled difficulty parameter:

| Case | \(\nu\) | Interpretation |
|---|---:|---|
| Easy | 0.05 | diffusion smooths the solution |
| Intermediate | 0.01 | stronger nonlinear steepening |
| Hard | \(0.01/\pi\approx0.003183\) | sharp internal layer |

Lower viscosity gives the network a progressively harder, higher-frequency spatial structure to learn.

## PINN

A tanh multilayer perceptron takes \((x,t)\) and outputs \(u_\theta(x,t)\). Automatic differentiation supplies

\[
 f_\theta = (u_\theta)_t + u_\theta (u_\theta)_x - \nu (u_\theta)_{xx}.
\]

The training objective is

\[
 \mathcal L = \mathcal L_{PDE}+10\mathcal L_{IC}+\mathcal L_{BC}.
\]

The same architecture, sampling points, weights, seed, and Adam phase are used in both PINN runs. The only controlled change is whether an L-BFGS refinement follows Adam.

---

# 4. Experiments

## P1 — Conventional baseline

A semi-implicit finite-volume method is used as the conventional solver. Convection uses a Rusanov numerical flux and diffusion is solved implicitly. A 1025-point solution is treated as the numerical reference; a 257-point solve is the lower-cost conventional baseline.

| \(\nu\) | FV-257 relative L2 error | FV-257 wall time |
|---:|---:|---:|
| 0.05 | 0.00570 | 0.0149 s |
| 0.01 | 0.01307 | 0.0188 s |
| 0.01/pi | 0.02311 | 0.0180 s |

The conventional baseline is both faster and more accurate than either PINN for this single forward solve, which is the expected outcome for a mature low-dimensional PDE discretization.

## P2 — Loss components

The code logs PDE-residual, initial-condition, and boundary-condition losses separately. The hard-case history is shown below.

![P2 loss components](figures/p2_loss_components_hard.png)

The L-BFGS stage continues reducing the composite objective after the fixed Adam budget has largely stalled. This is consistent with an optimization bottleneck rather than a network that simply cannot encode the solution.

## P3 — Error vs. difficulty knob

| \(\nu\) | FV baseline | Adam PINN | Adam + L-BFGS PINN | Error reduction from fix |
|---:|---:|---:|---:|---:|
| 0.05 | 0.00570 | 0.13905 | **0.01328** | **90.5%** |
| 0.01 | 0.01307 | 0.38351 | **0.15759** | **58.9%** |
| 0.01/pi | 0.02311 | 0.42541 | **0.28592** | **32.8%** |

![P3 error vs viscosity](figures/p3_error_vs_viscosity.png)

Two conclusions are visible. First, PINN accuracy collapses as viscosity decreases. Second, the optimizer fix helps at every viscosity but does not eliminate the low-viscosity failure.

## Controlled diagnosis — expressivity vs. optimization

The same neural architecture was trained directly on 5,000 reference solution samples for the hardest case. This supervised fit achieved

\[
\boxed{\text{relative L2 error}=0.0557}
\]

compared with 0.4254 for the Adam PINN and 0.2859 after L-BFGS.

This test separates two hypotheses:

1. **Architecture cannot represent the sharp solution.** The supervised result argues against this as the dominant explanation.
2. **Physics-informed optimization does not find the available representation.** The much larger PINN error and the improvement from L-BFGS support this explanation.

The remaining gap between the supervised fit and optimized PINN also suggests a secondary sharp-gradient/spectral-bias problem.

## P4 — Where is the error?

Adam-only pointwise error:

![Adam error field](figures/p4_error_field_adam.png)

Adam + L-BFGS pointwise error:

![Adam L-BFGS error field](figures/p4_error_field_adam_lbfgs.png)

The final-time comparison makes the mechanism clear: the basic PINN excessively smooths and misplaces the steep internal layer, while L-BFGS recovers a noticeably sharper transition.

![Hard-case final time](figures/hard_case_final_time.png)

The spatial error spectrum is also reported as required:

![Error spectrum](figures/p4_error_spectrum_hard.png)

High-frequency error remains after the optimizer fix, consistent with the fact that the shock-like layer is still under-resolved by the smooth tanh PINN.

## P5 — Cost

| \(\nu\) | Method | Training time | Inference time on 25,957 grid points |
|---:|---|---:|---:|
| 0.05 | Adam | 3.13 s | 0.0260 s |
| 0.05 | Adam + L-BFGS | 3.93 s | 0.0200 s |
| 0.01 | Adam | 3.38 s | 0.0121 s |
| 0.01 | Adam + L-BFGS | 3.90 s | 0.0109 s |
| 0.01/pi | Adam | 3.08 s | 0.0117 s |
| 0.01/pi | Adam + L-BFGS | 3.92 s | 0.0093 s |

![Training cost](figures/p5_training_cost.png)

These timings are from a CPU-only run with four PyTorch threads. They are hardware-dependent and should be rerun on the submission machine if timing is used quantitatively.

Because this study uses a PINN to solve one PDE instance rather than a trained neural operator reused over many inputs, a surrogate break-even query count \(N^*\) is not claimed.

---

# 5. Path C finding and recommendation

## Diagnosis

The baseline PINN failure is substantially an **optimization failure**:

- the same network can fit the hard reference directly to 5.57% error;
- Adam physics-informed training stops at 42.5% error;
- L-BFGS refinement lowers that error to 28.6% without changing architecture or training points.

## Fix

Use a two-stage optimizer:

1. Adam for robust early descent;
2. L-BFGS to refine the full-batch physics-informed objective.

This fix is targeted at the diagnosed optimization bottleneck rather than changing the PDE or giving the network interior labels.

## Recommendation

For a **single forward Burgers solve**, do not replace the finite-volume solver with a PINN. The conventional method is much more accurate and cheaper.

A PINN becomes more defensible when the actual application contains a feature that the conventional forward solve does not handle as naturally, such as sparse inverse measurements or an unknown coefficient. In that setting, the present experiment says that optimizer choice is not optional: Adam-only results should not be trusted without a stronger refinement stage and an independent numerical reference.

---

# 6. Assumptions and limitations

1. **Minimal PDE.** Burgers is a deliberately reduced model. Results should not be transferred quantitatively to a researcher's full CFD, heat-transfer, or multiphysics problem.
2. **One architecture.** The study isolates optimizer effects rather than searching architectures.
3. **Fixed loss weights.** The weights are held fixed in the controlled comparison. Adaptive weighting could provide an additional Path C experiment.
4. **Reference solver.** The included quantitative run uses an independently generated high-resolution finite-volume reference. The tiny public `burgers_shock.mat` file is supported as an external validation dataset when downloaded.
5. **Sharp-gradient limitation remains.** L-BFGS improves optimization but does not fully resolve the hard case. Fourier features, adaptive sampling, domain decomposition, or alternative activations are reasonable next experiments.
6. **Timing is machine-specific.** Accuracy conclusions are more portable than the reported wall-clock values.
7. **Researcher-specific value statement is pending.** The final memo must be rewritten in terms of the interviewed researcher's solver cost, accuracy threshold, and decision context.

---

# Reproducibility

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python src/download_data.py   # public external validation data
python run_all.py
```

Random seed and model settings are stored in `results/environment.json`. Raw metrics and losses are stored as CSV files in `results/`.

---

# References

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, 378, 2019.
2. Original PINN Burgers example and public reference data: `https://github.com/maziarraissi/PINNs`.
3. Public mirror used as an alternate data source: `https://github.com/XinyuanLiao/Adaptive-optimization-of-PINN`.
4. A. S. Krishnapriyan et al., “Characterizing possible failure modes in physics-informed neural networks,” NeurIPS 2021.
5. S. Wang, X. Yu, and P. Perdikaris, “When and why PINNs fail to train: A neural tangent kernel perspective,” *Journal of Computational Physics*, 449, 2022.
