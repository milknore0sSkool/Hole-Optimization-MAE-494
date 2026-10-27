# Researcher interview questions

Use these to connect the completed Burgers experiment to a real research problem.

1. What PDE or coupled equations do you solve, and what quantity do you ultimately care about?
2. What solver do you use now, and how long does one run take on what hardware?
3. How many solves do you need for a typical design study, calibration, UQ study, or control problem?
4. What makes the numerical problem hard: convection, thin layers, shocks, geometry, meshing, stiffness, multiple scales, or missing physics?
5. Is there a parameter that predictably makes the problem harder? That parameter should become our difficulty knob.
6. What measurements do you have, and how sparse/noisy are they?
7. What accuracy would be useful in practice?
8. Have you tried PINNs, neural operators, or other surrogates? If yes, where did they fail?
9. Would a reduced 1-D nonlinear transport problem with a sharp layer preserve the difficulty you care about, or would that simplify away the real issue?
10. What smallest result would make you want to test this further?
11. Who else in the group works most directly with the solver and should we talk to?
