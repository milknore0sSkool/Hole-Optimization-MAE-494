# Food Delivery Route Optimization (CVRP)

Optimization course project: formulating and solving a food-delivery routing
problem as a Capacitated Vehicle Routing Problem (CVRP).

- **[report.md](report.md)** — full write-up: problem formulation, decision
  variables, objective, constraints, classification, and assumptions.
- **`cvrp_milp_exact.m`** — exact MILP solver (MATLAB, requires Optimization
  Toolbox / `intlinprog`).
- **`cvrp_heuristic.m`** — nearest-neighbor + 2-opt heuristic for larger
  instances (plain MATLAB, no extra toolboxes).

## Running

Open either `.m` file in MATLAB and run it — each is self-contained, generates
its own random instance, solves it, prints the routes, and saves a plot
(`cvrp_exact_solution.png` / `cvrp_heuristic_solution.png`).

## Publishing to GitHub

```bash
git init
git add report.md cvrp_milp_exact.m cvrp_heuristic.m README.md
git commit -m "CVRP formulation and MATLAB solvers for food delivery routing"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Make sure the repo is set to **public** before submitting the link on Canvas.

## Before submitting

- [ ] Fill in team names / course / date at the top of `report.md`
- [ ] Run both MATLAB scripts and paste real output into the "Results and
      Interpretation" section of `report.md`
- [ ] Include the two generated plots (or push them to the repo) if you want
      figures in your presentation
- [ ] Double-check the math renders correctly on GitHub's Markdown preview
