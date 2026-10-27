# 7–8 minute presentation outline

## Slide 1 — Question

**Can optimizer choice rescue a PINN as nonlinear transport becomes convection dominated?**

- PDE: 1-D viscous Burgers
- Path C: diagnose failure + demonstrate fix
- Fix under test: Adam → L-BFGS

**Say:** We intentionally picked the smallest PDE that produces a sharp nonlinear layer so we could isolate optimization rather than hide behind a large CFD model.

## Slide 2 — Minimal case and difficulty knob

Show:

\[
u_t+u u_x-\nu u_{xx}=0
\]

- x in [-1,1], t in [0,1]
- u(x,0) = -sin(pi x)
- zero Dirichlet boundaries
- nu = 0.05, 0.01, 0.01/pi

**Say:** Lower viscosity makes the internal layer sharper and is our controlled difficulty knob.

## Slide 3 — Experimental design

Three comparisons:

1. finite-volume conventional baseline
2. PINN, Adam only
3. identical PINN, Adam followed by L-BFGS

Controlled quantities: architecture, collocation points, seed, and loss weights.

Diagnostic test: fit the same network directly to reference data.

## Slide 4 — P3 headline result

Use `figures/p3_error_vs_viscosity.png`.

Headline:

- Adam error: 13.9% → 42.5%
- Adam + L-BFGS: 1.33% → 28.6%
- optimizer fix helps everywhere but does not remove hard-case failure

## Slide 5 — Diagnosis

Use the supervised capacity result:

- hard-case Adam PINN: 42.5%
- hard-case Adam + L-BFGS: 28.6%
- same network, supervised: 5.57%

**Say:** The architecture can represent the field much better than the PINN objective finds. That points to optimization as a major cause, not just network size.

## Slide 6 — Where the failure occurs

Use `figures/hard_case_final_time.png` and, if time permits, `figures/p4_error_spectrum_hard.png`.

**Say:** The PINN smooths the internal layer. L-BFGS sharpens it, but high-frequency error remains. That is where a Fourier-feature or adaptive-sampling follow-up would target the model.

## Slide 7 — Cost and recommendation

- FV-257: ~0.015–0.019 s per solve in this run
- PINN training: ~3–4 s
- single forward solve: conventional solver wins

**Recommendation:** use PINNs only when the real problem has an inverse/sparse-data reason to justify them; do not use Adam alone on a sharp convection-dominated case.

## Slide 8 — Researcher connection / next experiment

Replace this slide with the actual interview:

- researcher's PDE and solver
- their difficulty parameter
- their useful accuracy threshold
- why the Burgers minimal case does or does not preserve their hard feature
- next experiment you recommend to them
