<p align="center">
  <img src="assets/hero_banner.png" alt="Project 3 — PINN Optimization" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PATH%20C-Diagnosis%20%2B%20Fix-ff4fa3?style=for-the-badge" alt="Path C">
  <img src="https://img.shields.io/badge/PDE-1D%20Burgers-d81b7d?style=for-the-badge" alt="1D Burgers equation">
  <img src="https://img.shields.io/badge/PyTorch-PINN-8b5cf6?style=for-the-badge" alt="PyTorch PINN">
  <img src="https://img.shields.io/badge/STATUS-Reproducible-5b214d?style=for-the-badge" alt="Reproducible">
</p>

<h3 align="center">A controlled study of when a physics-informed neural network fails — and how much a better optimizer can recover.</h3>

<p align="center">
  <a href="#-the-experiment">The experiment</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-diagnosis">Diagnosis</a> ·
  <a href="#-run-it">Run it</a> ·
  <a href="#-repository-map">Files</a>
</p>

---

## ✦ The one-line result

> **As viscosity decreases, the Burgers solution develops a sharp internal layer and an Adam-only PINN fails badly. Keeping the architecture, points, seed, and loss weights fixed — and changing only the optimization strategy to Adam → L-BFGS — cuts the error at every tested viscosity.**

<p align="center">
  <img src="assets/results_summary.png" alt="Results summary" width="100%">
</p>

---

## ♡ The experiment

We solve the one-dimensional viscous Burgers equation

$$
\boxed{u_t + u u_x - \nu u_{xx}=0}
$$

on

$$
x\in[-1,1], \qquad t\in[0,1],
$$

with

$$
u(x,0)=-\sin(\pi x), \qquad u(-1,t)=u(1,t)=0.
$$

The controlled **difficulty knob** is viscosity. Lower $\nu$ produces a thinner, sharper internal layer.

| Case | Viscosity $\nu$ | What changes |
|---|---:|---|
| 🌸 Easy | `0.05` | diffusion keeps the solution smooth |
| 💗 Intermediate | `0.01` | stronger nonlinear steepening |
| 🩷 Hard | `0.01/π ≈ 0.003183` | sharp internal layer / shock-like feature |

### Controlled comparison

Everything below is kept fixed between the two PINN runs:

- same tanh network architecture;
- same collocation, initial-condition, and boundary points;
- same random seed;
- same fixed loss weights;
- same Adam phase.

The **only intentional change** is whether an L-BFGS refinement stage follows Adam.

```mermaid
flowchart LR
    A["Finite-volume reference"] --> B["Adam PINN"]
    B --> C["Error grows as ν decreases"]
    C --> D["Same-network capacity test"]
    D --> E["Diagnosis: optimization bottleneck"]
    E --> F["Adam → L-BFGS"]
    F --> G["P1–P5 diagnostics"]

    style A fill:#fff1f7,stroke:#d81b7d,color:#321b2a
    style B fill:#ffe2f0,stroke:#ff4fa3,color:#321b2a
    style C fill:#ffd5e9,stroke:#ff4fa3,color:#321b2a
    style D fill:#f5eaff,stroke:#8b5cf6,color:#321b2a
    style E fill:#f5eaff,stroke:#8b5cf6,color:#321b2a
    style F fill:#eadcff,stroke:#8b5cf6,color:#321b2a
    style G fill:#fff1f7,stroke:#d81b7d,color:#321b2a
```

---

## ✦ Results

### Relative $L_2$ error

| $\nu$ | Finite volume | PINN · Adam | PINN · Adam → L-BFGS | Improvement from fix |
|---:|---:|---:|---:|---:|
| `0.05` | **0.57%** | 13.91% | **1.33%** | **90.5%** |
| `0.01` | **1.31%** | 38.35% | **15.76%** | **58.9%** |
| `0.01/π` | **2.31%** | 42.54% | **28.59%** | **32.8%** |

<p align="center">
  <img src="figures/p3_error_vs_viscosity.png" alt="Error versus viscosity" width="88%">
</p>

The optimizer fix helps throughout the sweep, but it does **not** make the hard case competitive with the conventional finite-volume solver.

### Hardest-case solution

<p align="center">
  <img src="figures/hard_case_final_time.png" alt="Hard-case final-time solution" width="88%">
</p>

The Adam-only PINN excessively smooths the steep transition. L-BFGS recovers a noticeably sharper profile, but residual high-frequency error remains.

---

## ✦ Diagnosis

The core question is whether the PINN fails because the network **cannot represent** the sharp solution or because the physics-informed objective is **hard to optimize**.

To separate those explanations, the same neural architecture was trained directly on reference values for the hardest case.

| Hard-case model | Relative $L_2$ error |
|---|---:|
| PINN · Adam | **42.54%** |
| PINN · Adam → L-BFGS | **28.59%** |
| Same network · supervised fit | **5.57%** |

> [!NOTE]
> The supervised capacity test is not presented as a replacement solver. It is a controlled diagnostic: the architecture can represent the solution much better than the physics-informed optimization is finding.

### What the evidence supports

1. **Optimization is a major failure mode.** L-BFGS improves the physics-informed solution without changing the architecture or training data.
2. **Expressivity alone is not the dominant limitation.** The same network reaches 5.57% when trained directly to the reference field.
3. **Sharp-gradient / spectral effects remain.** Even after L-BFGS, the lowest-viscosity case is still much worse than the finite-volume reference.

<table>
<tr>
<td width="50%" valign="top">
<strong>P2 · Loss components</strong><br><br>
<img src="figures/p2_loss_components_hard.png" alt="Loss components">
</td>
<td width="50%" valign="top">
<strong>P4 · Error spectrum</strong><br><br>
<img src="figures/p4_error_spectrum_hard.png" alt="Error spectrum">
</td>
</tr>
</table>

### Where the error lives

<table>
<tr>
<td width="50%" valign="top">
<strong>Adam only</strong><br><br>
<img src="figures/p4_error_field_adam.png" alt="Adam-only pointwise error">
</td>
<td width="50%" valign="top">
<strong>Adam → L-BFGS</strong><br><br>
<img src="figures/p4_error_field_adam_lbfgs.png" alt="Adam plus L-BFGS pointwise error">
</td>
</tr>
</table>

---

## ✦ P1–P5 checklist

| Requirement | Where it is handled |
|---|---|
| **P1 · Conventional baseline** | semi-implicit finite-volume solver in `src/numerics.py` |
| **P2 · Loss components** | `results/loss_history.csv` + `figures/p2_loss_components_hard.png` |
| **P3 · Error vs. difficulty knob** | viscosity sweep + `figures/p3_error_vs_viscosity.png` |
| **P4 · Error location / spectrum** | pointwise fields + FFT error spectrum |
| **P5 · Cost** | wall-clock training and inference measurements in `results/metrics.csv` |
| **Path C · Targeted fix** | Adam → L-BFGS, holding the rest of the experiment fixed |

<p align="center">
  <img src="figures/p5_training_cost.png" alt="Training cost" width="88%">
</p>

---

## ✦ Public data

The external validation file is the classic PINN Burgers dataset:

```text
burgers_shock.mat
```

It is approximately **203 KB** and contains `x`, `t`, and `usol`.

Canonical source:

```text
https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat
```

Download it automatically:

```bash
python src/download_data.py
```

### Python

```python
from scipy.io import loadmat

D = loadmat("data/burgers_shock.mat")
x = D["x"].squeeze()
t = D["t"].squeeze()
u = D["usol"]
```

### MATLAB

```matlab
D = load("data/burgers_shock.mat");
x = D.x(:);
t = D.t(:);
u = real(D.usol);
```

A ready-to-run MATLAB loader is included as [`load_burgers_data.m`](load_burgers_data.m).

---

## ✦ Run it

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Download the external reference dataset

```bash
python src/download_data.py
```

### 3. Reproduce the experiment

```bash
python run_all.py
```

Generated outputs go to:

```text
results/    raw metrics, predictions, loss histories
figures/    P2–P5 plots and solution comparisons
models/     saved PyTorch model weights
data/       reference fields and optional public .mat file
```

### Rebuild only the pink figures / README art

```bash
python src/make_plots.py
python src/make_readme_assets.py
```

---

## ✦ Repository map

```text
burgers_pinn_project/
│
├── assets/
│   ├── hero_banner.png
│   └── results_summary.png
│
├── data/                       # public + generated reference data
├── figures/                    # polished P2–P5 figures
├── models/                     # trained PyTorch weights
├── results/                    # CSV / NPZ raw outputs
│
├── src/
│   ├── data_utils.py
│   ├── download_data.py        # fetch burgers_shock.mat
│   ├── make_plots.py           # publication / README figures
│   ├── make_readme_assets.py   # banner + result cards
│   ├── numerics.py             # finite-volume baseline
│   ├── pinn.py                 # PINN + optimizer routines
│   └── run_experiments.py
│
├── burgers_pinn_project.ipynb  # notebook version
├── load_burgers_data.m         # MATLAB loader
├── PROJECT_REPORT.md           # full report
├── RECOMMENDATION_MEMO.md      # researcher-facing memo draft
├── OUTREACH_AND_INTERVIEW.md   # real interview fields
├── PRESENTATION_OUTLINE.md
├── SUBMISSION_CHECKLIST.md
├── requirements.txt
└── run_all.py
```

---

## ✦ Recommendation

For a **single forward Burgers solve**, the finite-volume method wins: it is faster and substantially more accurate.

The PINN becomes more defensible when the real application has something the conventional forward solve does not handle as naturally — for example an inverse coefficient, sparse measurements, or data/physics fusion. In that setting, this study says a basic Adam-only training run should **not** be treated as sufficient evidence that the method works or fails.

> [!IMPORTANT]
> **The researcher interview is intentionally not fabricated.** The computational study is complete, but the course requirement for a real researcher, problem brief, and researcher-specific memo must be completed using the actual interview. Use [`OUTREACH_AND_INTERVIEW.md`](OUTREACH_AND_INTERVIEW.md) and then tailor [`RECOMMENDATION_MEMO.md`](RECOMMENDATION_MEMO.md).

---

## ✦ Reproducibility

Current run configuration:

| Setting | Value |
|---|---|
| Random seed | `0` |
| Framework | PyTorch |
| Compute | CPU, 4 PyTorch threads |
| Adam epochs | `500` |
| Collocation points | `700` |
| L-BFGS iterations | `120` |
| Network | 3 hidden layers × 48 neurons |
| Activation | `tanh` |

Exact environment settings are stored in [`results/environment.json`](results/environment.json).

---

## ✦ References

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, 378, 2019.
2. A. S. Krishnapriyan et al., “Characterizing possible failure modes in physics-informed neural networks,” *NeurIPS*, 2021.
3. S. Wang, X. Yu, and P. Perdikaris, “When and why PINNs fail to train: A neural tangent kernel perspective,” *Journal of Computational Physics*, 449, 2022.
4. Original PINN Burgers code/data: `https://github.com/maziarraissi/PINNs`.

---

<p align="center">
  <b>Project 3 · PINN/PINO in the Field</b><br>
  <sub>Pretty in pink. Numerically suspicious until validated.</sub>
</p>
