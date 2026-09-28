# Project 2: Ill-Conditioning in Polynomial Thermocouple Calibration

## Abstract

This project studies numerical ill-conditioning in a thermocouple-inspired calibration problem. A synthetic dataset of thermocouple voltage and temperature measurements is fit using least-squares polynomial regression. The optimization problem belongs to **Family C: correlated / multiscale features**, because the monomial basis produces a Vandermonde design matrix whose columns become increasingly correlated as polynomial degree increases. The Hessian of the least-squares objective is \(H=X^T X\), so this correlation produces a rapidly growing condition number. The project first diagnoses the problem using the Hessian eigenvalue spectrum, the condition number \(\kappa(H)\), and the required diagonal-rescaling test. It then demonstrates the effect of ill-conditioning on gradient descent. Finally, the same polynomial model space is reparameterized using a Chebyshev basis, which reduces feature correlation and is expected to improve both the Hessian conditioning and gradient-descent convergence.

> **Status note.** This report is written before the final MATLAB run. Numerical result fields marked `TODO` should be replaced with the values and figures produced by the included MATLAB code.

---

## 1. Problem Identification and Motivation

Thermocouples are widely used for temperature measurement in mechanical and thermal experiments. The sensor produces a voltage that must be related to temperature through a calibration relationship. In an experimental setting, calibration data contain measurement noise and uncertainty, so a fitted model is often used to estimate temperature from measured voltage.

This project uses a **synthetic thermocouple-inspired calibration dataset** so that numerical conditioning can be studied without introducing uncontrolled experimental effects. The dataset contains 40 calibration points over a voltage range of 0 to 8 mV. A smooth nonlinear temperature response is used, and a small amount of measurement noise is added to represent calibration scatter.

The central question is:

> **How does increasing polynomial order in a thermocouple calibration model affect numerical conditioning and gradient-descent convergence, and can a Chebyshev basis improve the optimization without changing the underlying polynomial model space?**

This distinction is important in experimental mechanics. Measurement error and numerical ill-conditioning are not the same phenomenon. Measurement error describes uncertainty or scatter in the observed data, while numerical ill-conditioning describes sensitivity and poor curvature geometry in the optimization problem itself. This project focuses on the second issue.

---

## 2. Formulation

### 2.1 Calibration data

For each calibration point \(i=1,\ldots,N\), the dataset contains

\[
(V_i,T_i),
\]

where

- \(V_i\) is thermocouple voltage in mV,
- \(T_i\) is the measured calibration temperature in \(^\circ\mathrm C\),
- \(N=40\) is the number of calibration points.

To remove a trivial units/scale issue, voltage is normalized before polynomial fitting:

\[
z_i=\frac{V_i-V_{\mathrm{mid}}}{(V_{\max}-V_{\min})/2}.
\]

For the 0 to 8 mV range,

\[
V_{\mathrm{mid}}=4\ \mathrm{mV},
\]

so

\[
z_i=\frac{V_i-4}{4},
\qquad -1\le z_i\le 1.
\]

### 2.2 Decision variables

A degree-\(d\) polynomial calibration model in the monomial basis is

\[
\hat T(z)=a_0+a_1z+a_2z^2+\cdots+a_dz^d.
\]

The decision vector is

\[
\mathbf a=
\begin{bmatrix}
a_0 & a_1 & \cdots & a_d
\end{bmatrix}^T
\in\mathbb R^{d+1}.
\]

Because \(z\) is dimensionless, each coefficient contributes to a prediction measured in \(^\circ\mathrm C\).

### 2.3 Design matrix

The polynomial model can be written in matrix form as

\[
\hat{\mathbf T}=X\mathbf a,
\]

where the monomial/Vandermonde design matrix is

\[
X=
\begin{bmatrix}
1 & z_1 & z_1^2 & \cdots & z_1^d\\
1 & z_2 & z_2^2 & \cdots & z_2^d\\
\vdots & \vdots & \vdots & & \vdots\\
1 & z_N & z_N^2 & \cdots & z_N^d
\end{bmatrix}.
\]

### 2.4 Objective function

The polynomial coefficients are found by least squares:

\[
\boxed{
\min_{\mathbf a\in\mathbb R^{d+1}}
\;f(\mathbf a)
=\frac12\|X\mathbf a-\mathbf T\|_2^2
}
\]

where

\[
\mathbf T=
\begin{bmatrix}
T_1&T_2&\cdots&T_N
\end{bmatrix}^T.
\]

The residual vector is

\[
\mathbf r=\mathbf T-X\mathbf a.
\]

Calibration accuracy is also summarized using root-mean-square error,

\[
\mathrm{RMSE}
=\sqrt{\frac1N\sum_{i=1}^{N}(T_i-\hat T_i)^2}.
\]

### 2.5 Constraints and classification

There are no explicit constraints on the polynomial coefficients. Therefore, the problem is:

- continuous,
- unconstrained,
- smooth,
- convex,
- quadratic in the decision variables,
- a linear least-squares optimization problem.

---

## 3. Ill-Conditioning Mechanism

### 3.1 Family classification

This problem belongs to **Family C: correlated / multiscale features**.

The gradient of the least-squares objective is

\[
\nabla f(\mathbf a)=X^T(X\mathbf a-\mathbf T),
\]

and the Hessian is

\[
\boxed{H=X^T X}.
\]

The condition number of the Hessian is

\[
\boxed{
\kappa(H)=\frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
}
\]

for the positive-definite cases considered here.

As polynomial degree increases, the columns

\[
1,\;z,\;z^2,\;z^3,\ldots,z^d
\]

become increasingly correlated over the finite interval \([-1,1]\). High powers of \(z\) can become nearly linearly dependent, which causes one or more singular values of \(X\) to become small. Since

\[
H=X^T X,
\]

the Hessian eigenvalues are related to the squared singular values of \(X\). Small singular values therefore create very small Hessian eigenvalues and a large condition number.

### 3.2 Structural knob

The structural knob used in this project is the polynomial degree \(d\). The dataset is held fixed while the degree is increased:

\[
d=2,3,\ldots,12.
\]

The expected mechanism is

\[
 d\uparrow
 \quad\Rightarrow\quad
 \text{feature correlation}\uparrow
 \quad\Rightarrow\quad
 \lambda_{\min}(H)\downarrow
 \quad\Rightarrow\quad
 \kappa(H)\uparrow.
\]

### 3.3 Intrinsic-conditioning test

The required diagonal/Jacobi scaling test uses

\[
D=\operatorname{diag}(H)
\]

and

\[
\boxed{
H_s=D^{-1/2}HD^{-1/2}.
}
\]

If the large condition number were caused only by coordinate scale or units, diagonal rescaling would reduce \(\kappa\) to approximately order one. For this project, the ill-conditioning should remain large after this rescaling because the underlying problem is correlation between basis functions, not merely different coordinate magnitudes.

### 3.4 Small-case verification

For a low-order case, the code verifies the condition number two ways:

\[
\kappa_{\mathrm{eig}}
=\frac{\lambda_{\max}}{\lambda_{\min}}
\]

and MATLAB's

```matlab
cond(H)
```

These values should agree to numerical precision for the symmetric positive-definite Hessian.

---

## 4. Effect of Ill-Conditioning

### 4.1 D1 — Hessian spectrum and condition number

A representative high-order case, chosen here as degree \(d=10\), is used to plot the Hessian eigenvalues on a logarithmic scale.

The final results should report:

| Quantity | Result |
|---|---:|
| Polynomial degree | 10 |
| \(\lambda_{\min}(H)\) | TODO |
| \(\lambda_{\max}(H)\) | TODO |
| \(\kappa(H)\) | TODO |

![Hessian eigenvalue spectrum](fig_eigenvalue_spectrum.png)

A large spread between the smallest and largest eigenvalues indicates an elongated objective-function valley and therefore poor conditioning.

### 4.2 D2 — Condition number versus polynomial degree

The condition number is computed for each polynomial degree while the calibration dataset is held constant. The original Hessian and diagonally rescaled Hessian are plotted together.

![Condition number versus polynomial degree](fig_condition_vs_degree.png)

The expected result is that \(\kappa(H)\) increases rapidly with polynomial degree. If the scaled condition number also remains large, the intrinsic-conditioning requirement is satisfied.

Use the MATLAB output to complete the following representative table:

| Degree | \(\kappa(H)\) | \(\kappa(H_s)\) | RMSE (\(^\circ\mathrm C\)) |
|---:|---:|---:|---:|
| 2 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 6 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |
| 10 | TODO | TODO | TODO |
| 12 | TODO | TODO | TODO |

### 4.3 D3 — Baseline gradient descent

Gradient descent is used as the baseline first-order optimizer:

\[
\boxed{
\mathbf a_{k+1}
=\mathbf a_k-\alpha\nabla f(\mathbf a_k)
}
\]

with

\[
\nabla f(\mathbf a_k)=X^T(X\mathbf a_k-\mathbf T).
\]

For a strongly convex quadratic, the fixed step size is selected as

\[
\boxed{
\alpha=\frac{2}{L+\mu}
}
\]

where

\[
L=\lambda_{\max}(H),
\qquad
\mu=\lambda_{\min}(H).
\]

This choice avoids artificially making gradient descent slow through a poor step-size selection.

The convergence metric is the objective gap

\[
\boxed{f(\mathbf a_k)-f^*}
\]

shown on a semilogarithmic axis. The stopping criterion is based on the relative gradient norm,

\[
\frac{\|\nabla f(\mathbf a_k)\|_2}
{\|\nabla f(\mathbf a_0)\|_2}
\le 10^{-8}.
\]

A low-order and high-order monomial fit are compared to demonstrate the effect of increasing \(\kappa\).

![Gradient descent: low degree versus high degree](fig_gd_degree_effect.png)

Report the observed iteration counts:

| Monomial model | Iterations to tolerance |
|---|---:|
| Degree 3 | TODO |
| Degree 10 | TODO or `maxIter reached` |

If the degree-10 case reaches the maximum iteration count before satisfying the tolerance, that is reported directly rather than treated as a coding failure.

---

## 5. Proposed Solution and Demonstration

### 5.1 Chebyshev reparameterization

The ill-conditioning mechanism is caused by strong correlation among the monomial basis functions. Therefore, the remedy is to represent the same degree-\(d\) polynomial space using a basis that is closer to orthogonal over \([-1,1]\).

The Chebyshev basis is defined recursively by

\[
T_0(z)=1,
\]

\[
T_1(z)=z,
\]

and

\[
T_n(z)=2zT_{n-1}(z)-T_{n-2}(z).
\]

The calibration model becomes

\[
\hat T(z)
=c_0T_0(z)+c_1T_1(z)+\cdots+c_dT_d(z).
\]

The Chebyshev design matrix is

\[
X_C=
\begin{bmatrix}
T_0(z_1)&T_1(z_1)&\cdots&T_d(z_1)\\
T_0(z_2)&T_1(z_2)&\cdots&T_d(z_2)\\
\vdots&\vdots&&\vdots\\
T_0(z_N)&T_1(z_N)&\cdots&T_d(z_N)
\end{bmatrix},
\]

with Hessian

\[
H_C=X_C^T X_C.
\]

This is a **reparameterization**, not a different physical calibration problem. Both the monomial and Chebyshev formulations span the same degree-\(d\) polynomial space. The purpose is to improve the coordinates used by the optimizer.

### 5.2 D4 — Before/after conditioning

For the same degree-10 calibration problem, compare

\[
\kappa(H_{\mathrm{mono}})
\]

and

\[
\kappa(H_{\mathrm{Cheb}}).
\]

| Formulation | Condition number |
|---|---:|
| Monomial basis | TODO |
| Chebyshev basis | TODO |

The expected result is a substantial reduction in condition number because the Chebyshev columns are much less correlated over \([-1,1]\).

### 5.3 D4 — Before/after convergence

Gradient descent is then rerun using the same convergence tolerance and the optimal fixed step size for each Hessian.

![Gradient descent: monomial versus Chebyshev](fig_gd_monomial_vs_chebyshev.png)

Report the final comparison:

| Metric | Monomial basis | Chebyshev basis |
|---|---:|---:|
| Polynomial degree | 10 | 10 |
| \(\kappa(H)\) | TODO | TODO |
| GD iterations | TODO | TODO |
| Calibration RMSE (\(^\circ\mathrm C\)) | TODO | TODO |

The important comparison is that the calibration accuracy should remain similar because both formulations describe the same polynomial model space, while the condition number and optimization convergence can differ dramatically.

### 5.4 Calibration fit comparison

![Degree-10 calibration comparison](fig_calibration_comparison.png)

If both curves visually overlap while their condition numbers and iteration counts differ, this supports the conclusion that the improvement comes from numerical formulation rather than changing the physical calibration model.

---

## 6. Assumptions and Simplifications

The following assumptions are used:

1. **Synthetic calibration data.** The dataset is thermocouple-inspired rather than a manufacturer-specific thermocouple standard curve. This isolates the optimization behavior from uncontrolled experimental effects.
2. **Fixed dataset.** The same 40 calibration points are used for every polynomial degree so that polynomial degree is the primary structural knob.
3. **Normalized voltage.** Voltage is mapped to \([-1,1]\) before fitting. This intentionally removes a trivial unit-scale source of ill-conditioning.
4. **Independent measurement noise.** The synthetic calibration scatter is treated as independent measurement error and is not intended to represent every real thermocouple uncertainty source.
5. **Unweighted least squares.** All calibration points receive equal weight. A real experiment might use weighted least squares if uncertainty varies with temperature.
6. **No extrapolation study.** The project evaluates fitting and conditioning over the calibration interval only.
7. **No model-order selection claim.** High-order polynomials are used as a controlled mechanism for studying conditioning. The project does not claim that degree 10 or 12 is the preferred physical thermocouple calibration order.
8. **Gradient descent is a diagnostic baseline.** A direct least-squares solver is more efficient for this small problem, but gradient descent is used intentionally to demonstrate the effect of Hessian conditioning.

---

## 7. Discussion

The key distinction in this project is between **fit quality** and **optimization quality**. Increasing polynomial degree can leave RMSE nearly unchanged or even reduce it slightly while simultaneously making the Hessian much more ill-conditioned. Therefore, calibration accuracy alone does not reveal whether the optimization problem is numerically well formulated.

The monomial formulation is expected to become difficult because the basis functions become strongly correlated. This produces a Hessian with eigenvalues spanning many orders of magnitude. Gradient descent must use a step size small enough to remain stable in the largest-curvature direction, which causes very slow progress along the smallest-curvature direction.

The Chebyshev basis directly addresses this mechanism. Rather than adding regularization or changing the calibration data, it changes the coordinates used to represent the polynomial. A large reduction in \(\kappa\) together with faster convergence and similar RMSE would demonstrate that the original difficulty was largely a basis-conditioning problem.

`TODO after MATLAB run:` summarize the observed numerical trends here using the actual condition numbers and iteration counts.

---

## 8. Conclusion

This project formulates thermocouple calibration as a polynomial least-squares optimization problem and studies its numerical conditioning. The problem belongs to Family C because the monomial design matrix develops strongly correlated columns as polynomial degree increases. The Hessian \(H=X^TX\) therefore becomes increasingly ill-conditioned.

The final MATLAB results will be used to verify four claims:

1. The Hessian eigenvalue spectrum becomes increasingly spread as polynomial degree increases.
2. The condition number grows with polynomial degree and remains large after diagonal rescaling, demonstrating intrinsic rather than trivial unit-based ill-conditioning.
3. Gradient descent slows significantly as the condition number increases.
4. Reparameterizing the same polynomial space with a Chebyshev basis substantially reduces the condition number and improves gradient-descent convergence without materially changing calibration accuracy.

`TODO after MATLAB run:` replace this paragraph with a concise numerical conclusion using the final values of \(\kappa\), iteration counts, and RMSE.

---

## 9. Reproducibility

### 9.1 Files

Place the following files in the same MATLAB working directory:

```text
synthetic_thermocouple_calibration.csv
project2_thermocouple.m
```

The CSV file is the fixed synthetic dataset used for all reported results. Keeping the exact dataset in the public repository ensures that another user can reproduce the reported values.

### 9.2 MATLAB code

Save the following as `project2_thermocouple.m`.

```matlab
%% Project 2 - Ill-Conditioning in Thermocouple Calibration
clear;
clc;
close all;

%% 1. Import data

data = readtable('synthetic_thermocouple_calibration.csv');

V = data.Voltage_mV;
z = data.z_normalized;
T_true = data.T_true_C;
T_meas = data.T_measured_C;

N = length(V);

fprintf('---------------------------------------------\n');
fprintf('THERMOCOUPLE CALIBRATION DATA\n');
fprintf('---------------------------------------------\n');
fprintf('Number of calibration points: %d\n', N);
fprintf('Voltage range: %.2f to %.2f mV\n', min(V), max(V));
fprintf('Measured temperature range: %.2f to %.2f deg C\n\n', ...
    min(T_meas), max(T_meas));

%% 2. Synthetic calibration data

figure;
plot(V, T_true, 'LineWidth', 1.5);
hold on;
scatter(V, T_meas, 35, 'filled');
xlabel('Thermocouple Voltage (mV)');
ylabel('Temperature (^oC)');
title('Synthetic Thermocouple Calibration Data');
legend('True Response', 'Synthetic Measurements', 'Location', 'best');
grid on;
exportgraphics(gcf, 'fig_synthetic_data.png', 'Resolution', 300);

%% 3. Low-order calibration and small-case verification

d = 3;
X = buildMonomialMatrix(z, d);
a = X \ T_meas;
T_fit = X * a;
residuals = T_meas - T_fit;
RMSE = sqrt(mean(residuals.^2));

H = X' * X;
lambda = sort(eig((H + H')/2));
lambda_min = min(lambda);
lambda_max = max(lambda);
kappa_eigenvalues = lambda_max / lambda_min;
kappa_matlab = cond(H);

fprintf('---------------------------------------------\n');
fprintf('DEGREE-%d SMALL-CASE CHECK\n', d);
fprintf('---------------------------------------------\n');
fprintf('RMSE = %.6f deg C\n', RMSE);
fprintf('Kappa from eigenvalues = %.6e\n', kappa_eigenvalues);
fprintf('Kappa from cond(H)      = %.6e\n\n', kappa_matlab);

figure;
scatter(V, residuals, 35, 'filled');
hold on;
yline(0, '--');
xlabel('Thermocouple Voltage (mV)');
ylabel('Residual (^oC)');
title('Degree-3 Calibration Residuals');
grid on;
exportgraphics(gcf, 'fig_residuals_degree3.png', 'Resolution', 300);

%% 4. D2 - Degree sweep and diagonal rescaling

degrees = 2:12;
numDegrees = length(degrees);

kappa_monomial = zeros(numDegrees,1);
kappa_scaled = zeros(numDegrees,1);
RMSE_degree = zeros(numDegrees,1);

for i = 1:numDegrees
    d_i = degrees(i);
    X_i = buildMonomialMatrix(z, d_i);
    H_i = X_i' * X_i;

    kappa_monomial(i) = cond(H_i);

    dH = diag(H_i);
    D_inv_sqrt = diag(1 ./ sqrt(dH));
    H_scaled = D_inv_sqrt * H_i * D_inv_sqrt;
    kappa_scaled(i) = cond(H_scaled);

    a_i = X_i \ T_meas;
    T_fit_i = X_i * a_i;
    RMSE_degree(i) = sqrt(mean((T_meas - T_fit_i).^2));
end

figure;
semilogy(degrees, kappa_monomial, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
hold on;
semilogy(degrees, kappa_scaled, '-s', 'LineWidth', 1.5, 'MarkerSize', 7);
xlabel('Polynomial Degree');
ylabel('Condition Number, \kappa(H)');
title('Ill-Conditioning vs. Polynomial Degree');
legend('Original Hessian', 'After Diagonal Scaling', 'Location', 'northwest');
grid on;
exportgraphics(gcf, 'fig_condition_vs_degree.png', 'Resolution', 300);

conditioningTable = table(degrees', kappa_monomial, kappa_scaled, RMSE_degree, ...
    'VariableNames', {'Degree','Kappa_Monomial','Kappa_Scaled','RMSE_degC'});

disp('CONDITIONING RESULTS');
disp(conditioningTable);

%% 5. D1 - Hessian spectrum for representative ill-conditioned case

d_bad = 10;
X_bad = buildMonomialMatrix(z, d_bad);
H_bad = X_bad' * X_bad;
lambda_bad = sort(eig((H_bad + H_bad')/2), 'descend');

figure;
semilogy(1:length(lambda_bad), lambda_bad, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
xlabel('Eigenvalue Index');
ylabel('Hessian Eigenvalue');
title(sprintf('Hessian Eigenvalue Spectrum: Degree %d', d_bad));
grid on;
exportgraphics(gcf, 'fig_eigenvalue_spectrum.png', 'Resolution', 300);

fprintf('Degree-%d kappa(H) = %.6e\n\n', d_bad, cond(H_bad));

%% 6. D3 - Effect of degree on baseline gradient descent

maxIter = 100000;
tol = 1e-8;

% Low-order case
X_low = buildMonomialMatrix(z, 3);
H_low = X_low' * X_low;
ev_low = eig((H_low + H_low')/2);
mu_low = min(ev_low);
L_low = max(ev_low);
alpha_low = 2 / (L_low + mu_low);
a0_low = zeros(4,1);

[~, history_low, iterations_low] = ...
    gradientDescentLS(X_low, T_meas, a0_low, alpha_low, maxIter, tol);

% High-order case
X_high = buildMonomialMatrix(z, 10);
H_high = X_high' * X_high;
ev_high = eig((H_high + H_high')/2);
mu_high = min(ev_high);
L_high = max(ev_high);
alpha_high = 2 / (L_high + mu_high);
a0_high = zeros(11,1);

[a_GD, history_high, iterations_high] = ...
    gradientDescentLS(X_high, T_meas, a0_high, alpha_high, maxIter, tol);

figure;
semilogy(0:length(history_low)-1, history_low, 'LineWidth', 1.5);
hold on;
semilogy(0:length(history_high)-1, history_high, 'LineWidth', 1.5);
xlabel('Gradient Descent Iteration');
ylabel('Objective Gap, f(a_k)-f^*');
title('Effect of Polynomial Degree on Gradient Descent');
legend('Degree 3', 'Degree 10', 'Location', 'best');
grid on;
exportgraphics(gcf, 'fig_gd_degree_effect.png', 'Resolution', 300);

fprintf('---------------------------------------------\n');
fprintf('BASELINE GRADIENT DESCENT\n');
fprintf('---------------------------------------------\n');
fprintf('Degree 3 iterations  = %d\n', iterations_low);
fprintf('Degree 10 iterations = %d\n\n', iterations_high);

%% 7. D4 - Chebyshev remedy for degree 10

d_GD = 10;
X_mono = X_high;
H_mono = H_high;

X_cheb = buildChebyshevMatrix(z, d_GD);
H_cheb = X_cheb' * X_cheb;

kappa_mono = cond(H_mono);
kappa_cheb = cond(H_cheb);

ev_cheb = eig((H_cheb + H_cheb')/2);
mu_cheb = min(ev_cheb);
L_cheb = max(ev_cheb);
alpha_cheb = 2 / (L_cheb + mu_cheb);

c0 = zeros(d_GD + 1, 1);
[c_GD, history_cheb, iterations_cheb] = ...
    gradientDescentLS(X_cheb, T_meas, c0, alpha_cheb, maxIter, tol);

figure;
semilogy(0:length(history_high)-1, history_high, 'LineWidth', 1.5);
hold on;
semilogy(0:length(history_cheb)-1, history_cheb, 'LineWidth', 1.5);
xlabel('Gradient Descent Iteration');
ylabel('Objective Gap, f(x_k)-f^*');
title(sprintf('Degree-%d Gradient Descent: Monomial vs. Chebyshev', d_GD));
legend('Monomial Basis', 'Chebyshev Basis', 'Location', 'best');
grid on;
exportgraphics(gcf, 'fig_gd_monomial_vs_chebyshev.png', 'Resolution', 300);

%% 8. Final calibration comparison

T_fit_mono = X_mono * a_GD;
T_fit_cheb = X_cheb * c_GD;

RMSE_mono = sqrt(mean((T_meas - T_fit_mono).^2));
RMSE_cheb = sqrt(mean((T_meas - T_fit_cheb).^2));

figure;
scatter(V, T_meas, 35, 'filled');
hold on;
plot(V, T_fit_mono, 'LineWidth', 1.5);
plot(V, T_fit_cheb, '--', 'LineWidth', 1.5);
xlabel('Thermocouple Voltage (mV)');
ylabel('Temperature (^oC)');
title(sprintf('Degree-%d Thermocouple Calibration', d_GD));
legend('Synthetic Measurements', 'Monomial Basis', 'Chebyshev Basis', 'Location', 'best');
grid on;
exportgraphics(gcf, 'fig_calibration_comparison.png', 'Resolution', 300);

fprintf('---------------------------------------------\n');
fprintf('FINAL COMPARISON\n');
fprintf('---------------------------------------------\n');
fprintf('Monomial condition number   = %.6e\n', kappa_mono);
fprintf('Chebyshev condition number  = %.6e\n', kappa_cheb);
fprintf('Monomial GD iterations      = %d\n', iterations_high);
fprintf('Chebyshev GD iterations     = %d\n', iterations_cheb);
fprintf('Monomial RMSE               = %.6f deg C\n', RMSE_mono);
fprintf('Chebyshev RMSE              = %.6f deg C\n', RMSE_cheb);

summaryTable = table(kappa_mono, kappa_cheb, iterations_high, iterations_cheb, ...
    RMSE_mono, RMSE_cheb, ...
    'VariableNames', {'Kappa_Monomial','Kappa_Chebyshev', ...
    'Iterations_Monomial','Iterations_Chebyshev', ...
    'RMSE_Monomial','RMSE_Chebyshev'});

disp(summaryTable);

%% Local functions

function X = buildMonomialMatrix(z, d)
    N = length(z);
    X = zeros(N, d + 1);
    for j = 0:d
        X(:, j + 1) = z.^j;
    end
end

function X = buildChebyshevMatrix(z, d)
    N = length(z);
    X = zeros(N, d + 1);
    X(:,1) = 1;

    if d >= 1
        X(:,2) = z;
    end

    for j = 2:d
        X(:,j+1) = 2 .* z .* X(:,j) - X(:,j-1);
    end
end

function [x, history, iterations] = ...
    gradientDescentLS(X, y, x0, alpha, maxIter, tol)

    x = x0;

    % Direct solution is used only to define f* for the convergence plot.
    x_star = X \ y;
    residual_star = X*x_star - y;
    f_star = 0.5 * (residual_star' * residual_star);

    residual0 = X*x0 - y;
    gradient0 = X' * residual0;
    grad_ref = norm(gradient0);

    if grad_ref == 0
        grad_ref = 1;
    end

    history = zeros(maxIter + 1, 1);

    for k = 0:maxIter
        residual = X*x - y;
        f = 0.5 * (residual' * residual);
        gradient = X' * residual;

        history(k + 1) = max(f - f_star, eps);

        if norm(gradient) <= tol * grad_ref
            iterations = k;
            history = history(1:k+1);
            return;
        end

        x = x - alpha * gradient;
    end

    iterations = maxIter;
end
```

---

## 10. Final Checklist Before Submission

- [ ] Run the MATLAB script from a clean folder containing the CSV.
- [ ] Confirm the small-case eigenvalue ratio agrees with `cond(H)`.
- [ ] Replace all `TODO` values with actual MATLAB results.
- [ ] Confirm `fig_condition_vs_degree.png` demonstrates growth in \(\kappa\).
- [ ] Confirm diagonal scaling does **not** collapse \(\kappa\) to order one.
- [ ] Confirm the degree-10 Hessian spectrum spans a large range.
- [ ] Report whether degree-10 monomial GD converges or reaches `maxIter`.
- [ ] Compare the degree-10 monomial and Chebyshev condition numbers.
- [ ] Compare monomial and Chebyshev gradient-descent iteration counts.
- [ ] Confirm their RMSE values are similar enough to support the same-model-space argument.
- [ ] Add the CSV, MATLAB script, Markdown report, and generated PNG figures to the public GitHub repository.
- [ ] Verify all equations and images render correctly on GitHub.
