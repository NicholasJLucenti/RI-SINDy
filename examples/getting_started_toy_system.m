%  System being identified:
%      dx/dt = -k*x + alpha * Hill(x),   Hill(x) = 1 / (1 + (x/x0)^n)

%  x decays linearly (the drain term) and is produced by a repression
%  Hill function of itself (the regulatory term) 

%  Requires: Optimization Toolbox (lsqlin)
%            Signal Processing Toolbox (sgolayfilt)

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', 'src')));

k_true     = 0.5;    % linear decay rate
alpha_true = 3.0;    % Hill production strength
x0_true    = 2.0;    % Hill half-saturation point
n_true     = 4;       % Hill cooperativity

hill_func = @(x, x0, n) 1 ./ (1 + (x./x0).^n);

dt    = 0.05;
N     = 200;              % training window
t_all = (0:dt:(N+100)*dt)';

ode_rhs = @(t, x) -k_true*x + alpha_true*hill_func(x, x0_true, n_true);
[~, x_true] = ode45(@(t,x) ode_rhs(t,x), t_all, 0.1);

noise_level = 0.1;
rng(1);
x_data_all = x_true + noise_level*std(x_true)*randn(size(x_true));
x_data     = x_data_all(1:N);
t          = t_all(1:N);

% Numerical differentiation
dxdt = smooth_derivative(x_data_all, dt, 3, 11);
dxdt_tr = dxdt(1:N);

% Library construction
Theta_tr = [build_poly_library(x_data, 2), hill_func(x_data, x0_true, n_true)];
col_names = {'1', 'x', 'x^2', 'Hill(x)'};
pinned_col = 4;

%% --- 4. CONSTRAINTS -------------------------------------------------------
% Column order for the FREE columns only: [1, x, x^2] (Hill never
% appears here -- risindy.m never passes pinned columns to lsqlin).
%   - force the constant term to exactly zero (no constant production/decay)
%   - allow x and x^2 to be negative (both are decay-shaped terms)
lb{1} = [-inf, -inf, -inf];
ub{1} = [inf,   inf,    inf];

%% --- 5. DRIFT-BALANCE FUNCTION ---------------------------------------
% See drift_balance_toy below -- this is the ~10-line pattern every
% drift_balance_<system>.m file in paper/ follows.
drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
    drift_balance_toy(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                       x_data, x0_true, n_true);

%% --- 6. FIT ---------------------------------------------------------------
opts = struct();   % defaults are fine for a system this simple
[Xi, Xi_var, diagnostics] = risindy(Theta_tr, dxdt_tr, drift_fn, pinned_col, lb, ub, opts); %#ok<ASGLU>

threshold = 0.1;   % or whatever separates your real terms from the noise floor you're seeing
Xi(abs(Xi) < threshold) = 0;

%% --- 7. RESULTS -----------------------------------------------------------
fprintf('Identified equation:\n');
fprintf('  dx/dt = %+.4f  %+.4f * x  %+.4f * x^2  %+.4f * Hill(x)\n', Xi(1), Xi(2), Xi(3), Xi(4));
fprintf('Ground truth:\n');
fprintf('  dx/dt = %+.4f  %+.4f * x  %+.4f * x^2  %+.4f * Hill(x)\n\n', 0, -k_true, 0, alpha_true);

disp(table(Xi, sqrt(Xi_var), 'RowNames', col_names, 'VariableNames', {'Coefficient','StdDev'}));

%% --- 8. FORWARD INTEGRATION -----------------------------------------------
rhs  = @(xp) [poly_library_row(xp, 2), hill_func(xp, x0_true, n_true)] * Xi;
x_id = zeros(numel(t_all), 1);
x_id(1) = x_data_all(1);
for kk = 2:numel(t_all)
    x_id(kk) = x_id(kk-1) + dt * rhs(x_id(kk-1));
end

%% --- 9. PLOT --------------------------------------------------------------
figure('Color', 'w');
hold on;
fill([0, t(end), t(end), 0], [-1, -1, 3, 3], [0.85 0.85 0.85], ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(t_all, x_data_all, '.', 'Color', [0.6 0.6 0.6], 'MarkerSize', 6);
plot(t_all, x_true, 'k-', 'LineWidth', 1.2);
plot(t_all, x_id, 'b--', 'LineWidth', 1.5);
xlabel('Time'); ylabel('x');
legend({'Noisy data', 'True trajectory', 'RI-SINDy identified'}, 'Location', 'best');
title('RI-SINDy on a Minimal Self-Repressing Toy System');
box on; set(gca, 'Color', 'w', 'TickDir', 'out');


%% =========================================================================
%  DRIFT-BALANCE FUNCTION
%  Same ~10-line pattern as every paper/<system>/drift_balance_<system>.m
%  file: figure out which drain columns balance this regulatory term,
%  build the raw basis data, call drift_balance_generic.m, done. This
%  system only has ONE equation (vi is always 1), so there's no
%  switch/case on vi like the multi-equation paper systems have.
%% =========================================================================
function [pin_val, pin_var] = drift_balance_toy(XiN_smooth, XiN_var, vi, col_scale, target_scale, x_data, x0, n)
    drain_cols  = [2, 3];                             % x, x^2
    pinned_col  = 4;                                    % Hill(x)
    drain_basis = [x_data, x_data.^2];
    reg_basis   = 1 ./ (1 + (x_data./x0).^n);

    [pin_val, pin_var] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                drain_cols, pinned_col, drain_basis, reg_basis);
end