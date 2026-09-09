# Minimizing Food Delivery Routes: A Capacitated Vehicle Routing Formulation

**Team:** _[add your names here]_
**Course:** _[course name / number]_
**Date:** _[presentation date]_

---

## 1. Problem Identification and Motivation

A local restaurant runs its own delivery service instead of outsourcing to a third-party app. Every evening, dispatch has a batch of confirmed orders and a small fleet of riders (bikes, scooters, or cars), each of whom can only carry a limited number of orders at once before food starts getting cold or bags stop fitting on the rack. Dispatch has to decide which orders go to which rider and in what order they get dropped off.

This matters because delivery cost and speed are the main levers a small restaurant has to compete with larger delivery platforms: every extra kilometer driven is fuel, rider time, and a colder meal. Doing this by hand ("just send the closest orders together") tends to leave riders crossing paths or looping back across town, which is exactly the kind of inefficiency optimization is good at removing.

The problem is nontrivial for two reasons. First, it isn't just "visit everyone in the shortest order" (a Traveling Salesman Problem) — orders have to be split across multiple riders, and each rider has a hard capacity limit, so the *assignment* of orders to riders is itself a decision, not just the *sequencing*. Second, the number of ways to partition orders across riders and sequence each partition grows extremely fast, so even a modest number of orders makes brute-force enumeration impossible (see Section 5).

This is a standard formulation of the **Capacitated Vehicle Routing Problem (CVRP)**, applied here to a food-delivery context.

## 2. Decision Variables

Let node $0$ denote the restaurant (depot), and nodes $1, \dots, n$ denote the $n$ customer orders waiting to be delivered.

- $x_{ij} \in \{0,1\}$ for all $i, j \in \{0, 1, \dots, n\}$, $i \neq j$:
  equals $1$ if some rider travels directly from location $i$ to location $j$, and $0$ otherwise. **Binary.**

- $u_i \in \mathbb{R}$ for $i = 1, \dots, n$:
  the cumulative number of order-items a rider is carrying immediately after delivering to customer $i$, given that rider's route so far. **Continuous**, bounded by $d_i \le u_i \le Q$ (defined below). This is an auxiliary variable — it doesn't represent a physical decision by dispatch, but it's needed to make the formulation a linear program (see Section 4).

**Parameters** (data, not decisions):
- $c_{ij} \ge 0$: distance (km) between locations $i$ and $j$
- $d_i \ge 1$: number of items in customer $i$'s order, $i = 1, \dots, n$
- $Q$: maximum number of items a rider can carry per trip (capacity)
- $K$: number of riders available that shift

## 3. Objective Function

Minimize the total distance driven across all riders:

$$\min_{x} \quad \sum_{i=0}^{n} \sum_{\substack{j=0 \\ j \neq i}}^{n} c_{ij} \, x_{ij}$$

This is a **minimization** problem. Total distance is used as a proxy for fuel cost, rider time, and (indirectly) delivery speed, all of which move together for a fixed fleet.

## 4. Constraints

**Every customer is delivered to exactly once** (in-degree $=1$):

$$\sum_{\substack{i=0 \\ i \neq j}}^{n} x_{ij} = 1 \qquad \forall \, j = 1, \dots, n$$

**Every customer is left exactly once** (out-degree $=1$):

$$\sum_{\substack{j=0 \\ j \neq i}}^{n} x_{ij} = 1 \qquad \forall \, i = 1, \dots, n$$

Together these two force every customer to sit on exactly one rider's route, visited exactly once — no order is skipped or double-delivered.

**Exactly $K$ riders leave the depot, and exactly $K$ return:**

$$\sum_{j=1}^{n} x_{0j} = K, \qquad \sum_{i=1}^{n} x_{i0} = K$$

This fixes the fleet size actually used to the number of riders on shift, and — combined with the degree constraints above — guarantees every route both starts and ends at the restaurant.

**Capacity and subtour elimination** (Miller–Tucker–Zemlin style, adapted for capacity):

$$u_i - u_j + Q \, x_{ij} \le Q - d_j \qquad \forall \, i, j = 1, \dots, n, \; i \neq j$$

$$d_i \le u_i \le Q \qquad \forall \, i = 1, \dots, n$$

This single family of constraints does two jobs at once, which is why it looks less intuitive than the others:

- *Capacity:* if $x_{ij}=1$ (a rider goes straight from customer $i$ to customer $j$), the constraint forces $u_j \ge u_i + d_j$ — the running load after $j$ must be at least the running load after $i$ plus $j$'s items. Chained along a whole route, this means the load carried out of the depot must cover every item on that route, and since $u_i \le Q$ everywhere, no route can carry more than $Q$ items total.
- *Subtour elimination:* without these constraints, the degree constraints alone would technically be satisfied by a solution made of several small disconnected loops among customers that never touch the depot (e.g., customer 3 → 5 → 3), which is nonsense for an actual delivery plan. Because $u_i$ must strictly increase (by at least $d_j > 0$) around any cycle, no such loop can exist unless it passes through the depot (which has no $u$ variable and so isn't bound by this constraint) — that's what forces every route to be a simple path starting and ending at the restaurant.

**Binary restriction:**

$$x_{ij} \in \{0, 1\} \qquad \forall \, i \neq j$$

## 5. Problem Classification

This is a **Mixed-Integer Linear Program (MILP)** — specifically, the two-index vehicle-flow formulation of the **Capacitated Vehicle Routing Problem**, a well-studied **combinatorial optimization** problem.

- *Linear:* the objective and every constraint are linear in $x_{ij}$ and $u_i$.
- *Integer:* $x_{ij}$ is restricted to $\{0,1\}$; this integrality is what makes the feasible region nonconvex — its convex hull is a polytope, but the actual feasible set (lattice points satisfying all constraints) is a discrete set of routes, not a convex region.
- *NP-hard:* CVRP contains the Traveling Salesman Problem as the special case $K=1, Q=\infty$, so it inherits TSP's NP-hardness. The number of ways to partition $n$ customers across $K$ routes and sequence each route grows combinatorially, which is why exact solvers only scale to small/medium instances (Section 7) and real dispatch systems fall back on heuristics for anything larger.

## 6. Assumptions and Simplifications

- **Distance metric:** straight-line (Euclidean) distance is used as a proxy for actual road distance. Real streets, one-way restrictions, and traffic would change actual travel distance/time; a production system would use a road-network graph and real travel times instead.
- **Homogeneous fleet:** all $K$ riders have the same capacity $Q$ and the same (implicit) speed. In reality, a restaurant might mix bikes, scooters, and cars with different capacities and speeds.
- **Static, known demand:** all $n$ orders and their sizes are assumed known up front when routes are planned. Real delivery orders arrive dynamically throughout a shift, which would turn this into a *dynamic* or *online* routing problem.
- **No time windows:** the formulation ignores promised delivery times or food going cold — it only minimizes distance. A more realistic model for food specifically would add time-window constraints per customer, at the cost of a harder problem (VRPTW).
- **Single depot, single commodity:** all orders originate from one restaurant location; the model doesn't cover a chain with multiple kitchen locations.
- **Fixed fleet size $K$:** the number of riders is treated as a fixed shift parameter rather than a decision variable. An extension could let the model choose $K$ itself, trading off a fixed cost per rider against total distance.

---

## 7. Solution Methodology *(bonus)*

Two complementary implementations are provided in this repository:

- **`cvrp_milp_exact.m`** — solves the MILP formulation above exactly using MATLAB's `intlinprog` (Optimization Toolbox). This finds the *provably optimal* routing but only scales to a small number of customers ($n \lesssim 10$–$15$) before the combinatorial explosion makes exact solving impractical — this is exactly the NP-hardness from Section 5 showing up in practice.
- **`cvrp_heuristic.m`** — for larger, more realistic order volumes, uses a capacity-aware **nearest-neighbor construction** (greedily add the closest feasible customer to the current route, open a new rider's route when the current one is full) followed by **2-opt local search** on each route (repeatedly uncross any pair of edges that shortens the route). This is the kind of heuristic real dispatch software uses — it doesn't guarantee optimality, but it runs in a fraction of a second even for dozens of orders.

Both scripts generate a random synthetic instance (customer locations uniformly scattered around the restaurant, random order sizes), so they're fully self-contained and reproducible (fixed random seeds).

The formulation and index bookkeeping (mapping the $x_{ij}$ / $u_i$ variables into the flat vectors `intlinprog` expects) was cross-validated before writing the MATLAB code: an equivalent implementation was solved with a different MILP solver (SciPy/HiGHS) and checked against a brute-force optimal solution on a small 5-customer instance — the two matched exactly, confirming the constraint set is correct.

## 8. Results and Interpretation *(bonus)*

> Run `cvrp_milp_exact.m` and `cvrp_heuristic.m` in MATLAB and paste your actual output below before submitting — the numbers below are placeholders showing the format each script prints.

**Exact MILP** ($n=8$ customers, $K=3$ riders, $Q=6$ items/rider):

```
=== Exact MILP solution (intlinprog) ===
Optimal total distance: [FILL IN] km
Solve time: [FILL IN] s
Rider 1: [0 ...] | load _/6 | distance _ km
Rider 2: [0 ...] | load _/6 | distance _ km
Rider 3: [0 ...] | load _/6 | distance _ km
```

**Heuristic** ($n=30$ customers, $Q=6$ items/rider, riders opened as needed):

```
=== Heuristic solution (nearest neighbor + 2-opt) ===
Total distance: [FILL IN] km using [FILL IN] riders
Heuristic solve time: [FILL IN] s
```

**Interpretation notes to expand on once you have real numbers:**
- Compare the exact solution's total distance against what a naive "assign orders to the nearest rider in arrival order" baseline would give, to quantify the value of optimization.
- Report how 2-opt improves on the raw nearest-neighbor construction (the heuristic script prints both if you inspect `totalDist` before/after — see the comments in `cvrp_heuristic.m`).
- Note how solve time for the exact method changes if you increase $n$ in `cvrp_milp_exact.m` — this is a good way to demonstrate the NP-hardness claim from Section 5 empirically.
- Each script saves a plot (`cvrp_exact_solution.png`, `cvrp_heuristic_solution.png`) showing the routes on the map — include these in your presentation.

## 9. Code and Reproducibility *(bonus)*

- `cvrp_milp_exact.m` — exact MILP solver (requires MATLAB Optimization Toolbox for `intlinprog`)
- `cvrp_heuristic.m` — nearest-neighbor + 2-opt heuristic (no additional toolboxes required)

Both scripts are self-contained: run either one directly in MATLAB (no setup, no external data files) to regenerate the instance, solve it, print the routes, and save a plot. Random seeds (`rng(7)` and `rng(11)`) are fixed for reproducibility; change the parameters at the top of each script (`n`, `K`, `Q`, `areaKm`) to try different instance sizes.
