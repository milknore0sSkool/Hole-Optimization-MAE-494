%% Project 2 - Ill-Conditioning in Thermocouple Calibration
% Family C: Correlated / multiscale features
% Synthetic thermocouple-inspired polynomial calibration
%
% Baseline formulation: monomial polynomial basis
% Remedy: Chebyshev polynomial basis

clear;
clc;
close all;

%% 1. Import synthetic thermocouple calibration data

data = readtable('synthetic_thermocouple_calibration.csv');

V = data.Voltage_mV;        % Thermocouple voltage [mV]
z = data.z_normalized;      % Normalized voltage [-1,1]
T_true = data.T_true_C;     % Noise-free synthetic temperature [deg C]
T_meas = data.T_measured_C; % Synthetic measured temperature [deg C]

N = length(V);

fprintf('---------------------------------------------\n');
fprintf('THERMOCOUPLE CALIBRATION DATA\n');
fprintf('---------------------------------------------\n');
fprintf('Number of calibration points: %d\n', N);
fprintf('Voltage range: %.2f to %.2f mV\n', min(V), max(V));
fprintf('Measured temperature range: %.2f to %.2f deg C\n\n', ...
    min(T_meas), max(T_meas));

%% 2. Plot synthetic calibration data

figure;
plot(V, T_true, 'LineWidth', 1.5);
hold on;
scatter(V, T_meas, 35, 'filled');
xlabel('Thermocouple Voltage (mV)');
ylabel('Temperature (^oC)');
title('Synthetic Thermocouple Calibration Data');
legend('True Response', 'Synthetic Measurements', 'Location', 'best');
grid on;

%% 3. Initial low-order polynomial calibration

d = 3;
X = buildMonomialMatrix(z, d);

a = X \ T_meas;
T_fit = X * a;
residuals = T_meas - T_fit;
RMSE = sqrt(mean(residuals.^2));

fprintf('---------------------------------------------\n');
fprintf('DEGREE-%d CALIBRATION\n', d);
fprintf('---------------------------------------------\n');
fprintf('Polynomial coefficients:\n');
disp(a);
fprintf('RMSE = %.6f deg C\n\n', RMSE);

%% 4. Plot degree-3 calibration fit

figure;
scatter(V, T_meas, 35, 'filled');
hold on;
plot(V, T_fit, 'LineWidth', 1.5);
xlabel('Thermocouple Voltage (mV)');
ylabel('Temperature (^oC)');
title('Degree-3 Polynomial Calibration');
legend('Synthetic Measurements', 'Polynomial Fit', 'Location', 'best');
grid on;

%% 5. Plot calibration residuals

figure;
scatter(V, residuals, 35, 'filled');
hold on;
yline(0, '--');
xlabel('Thermocouple Voltage (mV)');
ylabel('Residual (^oC)');
title('Degree-3 Calibration Residuals');
grid on;

%% 6. D1 - Hessian and conditioning for degree-3 problem

H = X' * X;
lambda = sort(eig(H));

lambda_min = min(lambda);
lambda_max = max(lambda);

kappa_eigenvalues = lambda_max / lambda_min;
kappa_matlab = cond(H);

fprintf('---------------------------------------------\n');
fprintf('DEGREE-%d HESSIAN ANALYSIS\n', d);
fprintf('---------------------------------------------\n');
fprintf('Minimum eigenvalue       = %.6e\n', lambda_min);
fprintf('Maximum eigenvalue       = %.6e\n', lambda_max);
fprintf('Kappa from eigenvalues   = %.6e\n', kappa_eigenvalues);
fprintf('Kappa from cond(H)        = %.6e\n\n', kappa_matlab);

%% 7. D2 - Sweep polynomial degree and test intrinsic conditioning

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

    % Symmetric Jacobi / diagonal scaling
    D = diag(diag(H_i));
    D_inv_sqrt = diag(1 ./ sqrt(diag(D)));
    H_scaled = D_inv_sqrt * H_i * D_inv_sqrt;

    kappa_scaled(i) = cond(H_scaled);

    % Calibration accuracy
    a_i = X_i \ T_meas;
    T_fit_i = X_i * a_i;
    residual_i = T_meas - T_fit_i;
    RMSE_degree(i) = sqrt(mean(residual_i.^2));
end

%% 8. Plot condition number versus polynomial degree

figure;
semilogy(degrees, kappa_monomial, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
hold on;
semilogy(degrees, kappa_scaled, '-s', 'LineWidth', 1.5, 'MarkerSize', 7);
xlabel('Polynomial Degree');
ylabel('Condition Number, \kappa(H)');
title('Ill-Conditioning vs. Polynomial Degree');
legend('Original Hessian', 'After Diagonal Scaling', 'Location', 'northwest');
grid on;

%% 9. Calibration RMSE versus polynomial degree

figure;
plot(degrees, RMSE_degree, '-o', 'LineWidth', 1.5, 'MarkerSize', 7);
xlabel('Polynomial Degree');
ylabel('Calibration RMSE (^oC)');
title('Calibration Error vs. Polynomial Degree');
grid on;

%% 10. Conditioning results table

conditioningTable = table( ...
    degrees', ...
    kappa_monomial, ...
    kappa_scaled, ...
    RMSE_degree, ...
    'VariableNames', ...
    {'Degree','Kappa_Monomial','Kappa_Scaled','RMSE_degC'});

disp(' ');
disp('CONDITIONING RESULTS');
disp(conditioningTable);

%% 11. D1 - Hessian eigenvalue spectrum for an ill-conditioned case

d_bad = 10;
X_bad = buildMonomialMatrix(z, d_bad);
H_bad = X_bad' * X_bad;
lambda_bad = sort(eig(H_bad), 'descend');

figure;
semilogy(1:length(lambda_bad), lambda_bad, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7);
xlabel('Eigenvalue Index');
ylabel('Hessian Eigenvalue');
title(sprintf('Hessian Eigenvalue Spectrum: Degree %d', d_bad));
grid on;

%% 12. D3 - Baseline gradient descent using monomial basis

d_GD = 10;

X_GD = buildMonomialMatrix(z, d_GD);
H_GD = X_GD' * X_GD;

lambda_GD = eig(H_GD);
lambda_min_GD = min(lambda_GD);
lambda_max_GD = max(lambda_GD);

% Optimal fixed step size for an SPD quadratic
alpha_GD = 2 / (lambda_max_GD + lambda_min_GD);

a0 = zeros(d_GD + 1, 1);
maxIter = 100000;
tol = 1e-6;

[a_GD, history_mono, iterations_mono] = ...
    gradientDescentLS(X_GD, T_meas, a0, alpha_GD, maxIter, tol);

%% 13. D4 - Chebyshev basis remedy

X_cheb = buildChebyshevMatrix(z, d_GD);
H_cheb = X_cheb' * X_cheb;
kappa_cheb = cond(H_cheb);

fprintf('---------------------------------------------\n');
fprintf('MONOMIAL VS CHEBYSHEV CONDITIONING\n');
fprintf('---------------------------------------------\n');
fprintf('Polynomial degree: %d\n', d_GD);
fprintf('Monomial condition number   = %.6e\n', cond(H_GD));
fprintf('Chebyshev condition number  = %.6e\n\n', kappa_cheb);

%% 14. Gradient descent using Chebyshev basis

lambda_cheb = eig(H_cheb);
lambda_min_cheb = min(lambda_cheb);
lambda_max_cheb = max(lambda_cheb);

alpha_cheb = 2 / (lambda_max_cheb + lambda_min_cheb);

c0 = zeros(d_GD + 1, 1);

[c_GD, history_cheb, iterations_cheb] = ...
    gradientDescentLS(X_cheb, T_meas, c0, alpha_cheb, maxIter, tol);

%% 15. D4 - Compare gradient-descent convergence

figure;
semilogy(0:length(history_mono)-1, history_mono, 'LineWidth', 1.5);
hold on;
semilogy(0:length(history_cheb)-1, history_cheb, 'LineWidth', 1.5);
xlabel('Gradient Descent Iteration');
ylabel('Objective Gap, f(x_k) - f^*');
title(sprintf('Gradient Descent Convergence: Degree %d', d_GD));
legend('Monomial Basis', 'Chebyshev Basis', 'Location', 'best');
grid on;

%% 16. Compare iteration counts

fprintf('---------------------------------------------\n');
fprintf('GRADIENT DESCENT COMPARISON\n');
fprintf('---------------------------------------------\n');
fprintf('Tolerance = %.1e relative initial gradient norm\n', tol);
fprintf('Monomial iterations  = %d\n', iterations_mono);
fprintf('Chebyshev iterations = %d\n\n', iterations_cheb);

%% 17. Compare final calibration curves and RMSE

T_fit_mono = X_GD * a_GD;
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

fprintf('Final monomial RMSE   = %.6f deg C\n', RMSE_mono);
fprintf('Final Chebyshev RMSE  = %.6f deg C\n', RMSE_cheb);

%% 18. Final summary table

summaryTable = table( ...
    cond(H_GD), ...
    cond(H_cheb), ...
    iterations_mono, ...
    iterations_cheb, ...
    RMSE_mono, ...
    RMSE_cheb, ...
    'VariableNames', ...
    {'Kappa_Monomial', ...
     'Kappa_Chebyshev', ...
     'Iterations_Monomial', ...
     'Iterations_Chebyshev', ...
     'RMSE_Monomial', ...
     'RMSE_Chebyshev'});

disp(' ');
disp('FINAL COMPARISON');
disp(summaryTable);

%% ================================================================
% LOCAL FUNCTIONS
% ================================================================

function X = buildMonomialMatrix(z, d)
%BUILDMONOMIALMATRIX Build X = [1 z z^2 ... z^d].

    N = length(z);
    X = zeros(N, d + 1);

    for j = 0:d
        X(:, j + 1) = z.^j;
    end
end

function X = buildChebyshevMatrix(z, d)
%BUILDCHEBYSHEVMATRIX Build Chebyshev design matrix using recurrence.
% T0(z) = 1
% T1(z) = z
% Tn(z) = 2*z*T_(n-1)(z) - T_(n-2)(z)

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
%GRADIENTDESCENTLS Gradient descent for
%   f(x) = 1/2 ||X*x - y||^2
%
% Stops when
%   ||grad f(x_k)|| <= tol * ||grad f(x_0)||
%
% history stores the requested diagnostic quantity
%   f(x_k) - f^*.

    x = x0;

    % Direct least-squares optimum used only as a reference
    x_star = X \ y;
    residual_star = X*x_star - y;
    f_star = 0.5 * (residual_star' * residual_star);

    % Reference gradient norm for relative stopping criterion
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

        % D3 convergence metric
        objectiveGap = max(f - f_star, eps);
        history(k + 1) = objectiveGap;

        % Relative-gradient stopping condition
        if norm(gradient) <= tol * grad_ref
            iterations = k;
            history = history(1:k+1);
            return;
        end

        % Gradient descent update
        x = x - alpha * gradient;
    end

    iterations = maxIter;
end
