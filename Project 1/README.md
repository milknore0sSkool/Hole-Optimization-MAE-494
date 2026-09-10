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

Open the `.m` file in MATLAB and run it, generates
its own random instance, solves it, prints the routes, and saves a plot
(`cvrp_exact_solution.png` / `cvrp_heuristic_solution.png`).
