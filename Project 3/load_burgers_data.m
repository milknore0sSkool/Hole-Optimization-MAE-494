% Load the public classic Burgers reference dataset in MATLAB.
% First place burgers_shock.mat in ./data/ (see data/README.md).
D = load(fullfile('data','burgers_shock.mat'));
x = D.x(:);
t = D.t(:);
u = real(D.usol);          % convention: x-by-time

fprintf('x: %d points, t: %d points, usol: %d x %d\n', ...
    numel(x), numel(t), size(u,1), size(u,2));

imagesc(t, x, u);
axis xy;
xlabel('t'); ylabel('x');
title('Burgers reference solution');
colorbar;
