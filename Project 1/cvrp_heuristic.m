%% Food Delivery Routing — Heuristic for Larger Instances
% Capacity-aware nearest-neighbor construction + 2-opt local search.
% The exact MILP (cvrp_milp_exact.m) doesn't scale past a few dozen
% customers; this heuristic is what a real dispatch system would use.
% Unlike the exact script, the number of riders is not fixed in advance —
% the construction heuristic opens a new route whenever the current one
% is full.
%
% No toolboxes required (plain MATLAB).

clear; clc; close all;
rng(11);

%% ---- Problem parameters ----
n      = 30;   % number of customer orders (try bumping this up — the
               % exact MILP would choke on an instance this size)
Q      = 6;    % max orders a rider can carry per trip
areaKm = 10;

%% ---- Generate instance ----
depot     = [areaKm/2, areaKm/2];
customers = areaKm * rand(n, 2);
demand    = randi([1 3], n, 1);
coords    = [depot; customers];
N = n + 1;

C = zeros(N, N);
for i = 1:N
    for j = 1:N
        C(i,j) = norm(coords(i,:) - coords(j,:));
    end
end

%% ---- Construction: capacity-aware nearest neighbor ----
unvisited = 2:N;
routes = {};
tic;
while ~isempty(unvisited)
    route = 1;      % start at depot
    load  = 0;
    cur   = 1;
    while true
        candidates = unvisited(demand(unvisited - 1) <= Q - load);
        if isempty(candidates)
            break;
        end
        [~, k] = min(C(cur, candidates));
        nextNode = candidates(k);
        route(end+1) = nextNode; %#ok<AGROW>
        load = load + demand(nextNode - 1);
        unvisited(unvisited == nextNode) = [];
        cur = nextNode;
    end
    route(end+1) = 1;
    routes{end+1} = route; %#ok<AGROW>
end

%% ---- Improvement: 2-opt within each route ----
for r = 1:numel(routes)
    routes{r} = twoOpt(routes{r}, C);
end
heurTime = toc;

%% ---- Report ----
totalDist = 0;
fprintf('\n=== Heuristic solution (nearest neighbor + 2-opt) ===\n');
for r = 1:numel(routes)
    route = routes{r};
    routeDist = 0;
    for k = 1:numel(route)-1
        routeDist = routeDist + C(route(k), route(k+1));
    end
    routeDemand = sum(demand(route(2:end-1) - 1));
    totalDist = totalDist + routeDist;
    fprintf('Rider %d: %s | load %d/%d | distance %.2f km\n', ...
        r, mat2str(route - 1), routeDemand, Q, routeDist);
end
fprintf('Total distance: %.3f km using %d riders\n', totalDist, numel(routes));
fprintf('Heuristic solve time: %.4f s\n', heurTime);

%% ---- Plot ----
figure; hold on; axis equal;
colors = lines(numel(routes));
plot(depot(1), depot(2), 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
text(depot(1)+0.1, depot(2)+0.1, 'Restaurant');
for c = 1:n
    plot(customers(c,1), customers(c,2), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [.7 .7 .7]);
end
for r = 1:numel(routes)
    rc = coords(routes{r}, :);
    plot(rc(:,1), rc(:,2), '-', 'Color', colors(r,:), 'LineWidth', 1.8);
end
title(sprintf('Food Delivery CVRP — Heuristic Solution (n=%d, Total = %.2f km, %d riders)', ...
    n, totalDist, numel(routes)));
xlabel('km'); ylabel('km'); grid on;
saveas(gcf, 'cvrp_heuristic_solution.png');

%% ---- Local function: 2-opt ----
function route = twoOpt(route, C)
    improved = true;
    while improved
        improved = false;
        for i = 2:numel(route)-2
            for j = i+1:numel(route)-1
                a = route(i-1); b = route(i);
                c = route(j);   d = route(j+1);
                delta = (C(a,c) + C(b,d)) - (C(a,b) + C(c,d));
                if delta < -1e-9
                    route(i:j) = route(j:-1:i);
                    improved = true;
                end
            end
        end
    end
end
