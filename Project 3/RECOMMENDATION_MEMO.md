<p align="center">
  <img src="assets/results_summary.png" alt="PINN optimization results" width="100%">
</p>

<p align="center"><b>RESEARCHER RECOMMENDATION MEMO</b></p>

---

# Recommendation memo — PINNs for a convection-dominated PDE

**To:** [Researcher name]  
**From:** [Team members]  
**Subject:** Recommendation on PINN training for sharp nonlinear transport  
**Date:** [Date]

## Recommendation

For a single forward solve, I would **not recommend replacing your conventional PDE solver with a basic PINN**. In our controlled Burgers-equation case, the conventional finite-volume method was both more accurate and cheaper. I would consider a PINN only if your actual problem has an inverse or sparse-data component that gives the physics-informed formulation a capability advantage over a standard forward solver.

## What we tested

We used the 1-D viscous Burgers equation as the smallest problem that retains nonlinear convection and a sharp internal layer. We decreased viscosity from 0.05 to 0.01/pi to make the problem progressively harder. We compared a conventional finite-volume reference, an Adam-only PINN, and an otherwise identical PINN refined with L-BFGS.

## Main result

The Adam-only PINN error increased from 13.9% to 42.5% as viscosity decreased. Adding L-BFGS reduced error to 1.33%, 15.8%, and 28.6% across the three cases. In the hardest case, the same neural network fitted directly to solution data reached 5.57% error. That gap indicates that the architecture has substantially more representational capacity than the physics-informed optimization is able to exploit.

Our conclusion is therefore that **optimization is a major failure mechanism**, while the remaining error at the sharp layer indicates that high-frequency representation is also a limitation.

## What this means for your problem

If your model contains strong convection, thin layers, fronts, or other sharp spatial features, an Adam-only PINN result should be treated cautiously even when the training loss looks small. At minimum, I would use an independent conventional reference on a reduced case and add a quasi-Newton refinement such as L-BFGS.

## Next experiment

The most useful next experiment would be to reproduce one reduced version of your actual PDE with the same difficulty parameter that controls the sharp feature. I would then compare:

1. your current conventional solver;
2. Adam → L-BFGS PINN;
3. the same PINN with a sharp-feature remedy such as Fourier features or residual-adaptive sampling.

The decision criterion should be the accuracy threshold that matters for your research, not PINN training loss alone.

**Confidence:** moderate for the optimization diagnosis on this minimal case; low for direct transfer to your full problem until the reduced case is matched to your geometry, parameters, and accuracy requirements.
