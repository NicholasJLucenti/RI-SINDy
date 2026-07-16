%% =========================================================================
%  NF-kB: Posterior Standard Deviations Under Three Priors
%
%  Fills a gap not covered by nfkb_risindy.m or nfkb_comparisons.m:
%  neither script runs RI-SINDy under more than one sparsity prior.
%  This script reuses the exact same data generation, library, bounds,
%  and drift-balance function as nfkb_risindy.m, but loops risindy.m
%  over 'SpikeSlab', 'Laplace', and 'Horseshoe', printing a table that
%  matches paper Table "uq_ikb_priors" directly: posterior standard
%  deviations for the I-kB equation (dimension j=2) under each prior.
%
%  Requires: Optimization Toolbox (lsqlin)
%            Signal Processing Toolbox (sgolayfilt)
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

%% --- HYPERPARAMETERS (identical to nfkb_risindy.m) -------------------------
polyorder = 3;
dt        = 0.0375;
N         = 800;
hill_n    = 3;
hill_k0   = 1.5;

hill_act = @(u, k, n)  u.^n ./ (k^n + u.^n);

%% --- GROUND-TRUTH PARAMETERS & SYNTHETIC DATA (identical to nfkb_risindy.m) ---
alpha_z = 8.0;
kxy     = 1.2;
kx      = 0.4;
ky_z    = 1.8;
ky      = 0.3;
kz      = 0.8;

t_end     = (N + 200) * dt;
t_span    = 0:dt:t_end;
signal_fn = @(t) 3.5 * exp(-0.15*t) .* (t > 0);

ode_rhs = @(t, s) [ ...
    signal_fn(t)  -  kxy*s(1)*s(2)  -  kx*s(1); ...
    ky_z*s(3)     -  kxy*s(1)*s(2)  -  ky*s(2); ...
    alpha_z * hill_act(s(1), hill_k0, hill_n)  -  kz*s(3) ];

opts_ode   = odeset('RelTol',1e-9,'AbsTol',1e-11);
[t_ode, S] = ode45(ode_rhs, t_span, [0.1; 0.05; 0.02], opts_ode);

x_data_all = S(:,1);
y_data_all = S(:,2);
z_data_all = S(:,3);

noise_level = 0.1;
rng(42);   % same seed as nfkb_risindy.m -- reproduces the identical noisy dataset
x_data_all = x_data_all + noise_level*std(x_data_all)*randn(size(x_data_all));
y_data_all = y_data_all + noise_level*std(y_data_all)*randn(size(y_data_all));
z_data_all = z_data_all + noise_level*std(z_data_all)*randn(size(z_data_all));

x_data = x_data_all(1:N);
y_data = y_data_all(1:N);
z_data = z_data_all(1:N);
t      = t_ode(1:N);
signal_train = signal_fn(t);

%% --- DERIVATIVES (identical to nfkb_risindy.m) -----------------------------
dxdt_all = smooth_derivative(x_data_all, dt, 3, 11);
dydt_all = smooth_derivative(y_data_all, dt, 3, 11);
dzdt_all = smooth_derivative(z_data_all, dt, 3, 11);
dXdt_tr  = [dxdt_all(1:N), dydt_all(1:N), dzdt_all(1:N)];

%% --- LIBRARY (identical to nfkb_risindy.m) ---------------------------------
col_names = {'1','S','x','x^2','x^3','y','y^2','y^3','z','z^2','z^3','x*y','Hill(x)'};

HillAct_x = hill_act(x_data, hill_k0, hill_n);
Theta_tr = [ ones(N,1), signal_train, ...
             x_data, x_data.^2, x_data.^3, ...
             y_data, y_data.^2, y_data.^3, ...
             z_data, z_data.^2, z_data.^3, ...
             x_data.*y_data, HillAct_x ];

%% --- CONSTRAINTS (identical to nfkb_risindy.m) -----------------------------
lb{1} = [0, 0, -inf, -inf, -inf, -inf, -inf, -inf,   0,   0,   0, -inf];
ub{1} = [0, inf,   0,    0,    0,    inf,    inf,    inf,   inf,   inf,   inf,    0];
lb{2} = [-inf, 0,    0,    0,    0, -inf, -inf, -inf,   0,   0,   0, -inf];
ub{2} = [inf, 0, inf,  inf,  inf,    0,    0,    0,   inf,   inf,   inf,    0];
lb{3} = [-inf, 0, -inf, -inf, -inf,   0,   0,   0, -inf, -inf, -inf,    0];
ub{3} = [inf, 0,   0,    0,    0,   0,   0,   0,    inf,    inf,    inf,    0];

pinned_col = 13;   % Hill(x)

drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
    drift_balance_nfkb(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                        x_data, z_data, hill_k0, hill_n);

%% --- FIT UNDER EACH PRIOR ---------------------------------------------------
% prior_params intentionally omitted below -- risindy.m auto-fills the
% correct default parameters per prior (see parse_opts in risindy.m),
% matching what each prior would use if run standalone.
priors = {'SpikeSlab', 'Laplace', 'Horseshoe'};
priorLabels = {'SS', 'LAP', 'RH'};   % matches paper table column headers

Xi_all     = cell(1,3);
Xi_std_all = cell(1,3);

for p = 1:numel(priors)
    fprintf('\n--- Fitting NF-kB under %s prior ---\n', priors{p});
    opts = struct();
    opts.eta_pin     = 0.6;
    opts.eta_weight  = 0.7;
    opts.eta_drain   = 0;
    opts.n_iter      = 20;
    opts.w_threshold = 2;
    opts.prior       = priors{p};

    [Xi, Xi_var] = risindy(Theta_tr, dXdt_tr, drift_fn, pinned_col, lb, ub, opts);

    threshold = 0.2;
    Xi(abs(Xi) < threshold) = 0;

    Xi_all{p}     = Xi;
    Xi_std_all{p} = sqrt(Xi_var);
end

%% --- PRINT: coefficients under each prior (sanity check across priors) ----
for p = 1:numel(priors)
    print_addpath = fullfile(thisDir, '..', 'comparisons', 'utils');
    if ~any(strcmp(strsplit(path,pathsep), print_addpath))
        addpath(print_addpath);
    end
    print_coeffs(sprintf('RI-SINDy [%s]', priors{p}), Xi_all{p}, col_names, ...
                 {'dx/dt','dy/dt','dz/dt'});
end

%% --- PRINT: Table matching paper's "uq_ikb_priors" (I-kB, dimension j=2) --
fprintf('\n=========================================================\n');
fprintf(' Posterior std, I-kB equation (dim j=2), by prior\n');
fprintf(' (paste directly into the paper table)\n');
fprintf('=========================================================\n');
fprintf('%-10s %10s %10s %10s\n', 'Term', priorLabels{1}, priorLabels{2}, priorLabels{3});
for j = 1:numel(col_names)
    fprintf('%-10s %10.4f %10.4f %10.4f\n', col_names{j}, ...
        Xi_std_all{1}(j,2), Xi_std_all{2}(j,2), Xi_std_all{3}(j,2));
end
fprintf('=========================================================\n');