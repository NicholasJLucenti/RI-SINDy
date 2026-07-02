%% =========================================================================
%  NF-kB Inflammatory Signaling Network -- RI-SINDy fit
%
%  State variables: x = nuclear NF-kB, y = IkBa protein, z = IkBa mRNA.
%  Produces the RI-SINDy coefficients used throughout the paper: fits
%  the sparse polynomial drain/production terms via risindy.m while the
%  Hill activation term (HillAct(x), z-equation only) is pinned by
%  drift_balance_nfkb.m rather than fit by regression.
%
%  Requires: Optimization Toolbox (lsqlin)
%            Signal Processing Toolbox (sgolayfilt)
%
%  STRUCTURAL NOTE: the original script used TWO different library
%  matrices (Theta_xy for the x/y equations, Theta_z for the z
%  equation), reusing the same column positions 6-8 for "y" in one and
%  "z" in the other. risindy.m needs ONE shared library across all
%  equations, so this version unifies them into a single 13-column
%  Theta with dedicated y and z columns, and uses lb/ub bounds to force
%  each equation's inapplicable columns (e.g. z-terms in the x/y
%  equations) to exactly zero -- functionally identical to never
%  including those columns in the first place, and the same pattern
%  already used for Goodwin's "every equation sees the full library"
%  approach.
%
%  Two other deliberate deviations from the literal original script are
%  documented in drift_balance_nfkb.m: the 5% force multiplier is kept,
%  but the avg_drain sign convention now uses mean(abs(...)) instead of
%  abs(mean(...)), matching the correction already applied project-wide.
% =========================================================================

close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

%% --- HYPERPARAMETERS ------------------------------------------------------
polyorder = 3;
dt        = 0.0375;
N         = 800;
hill_n    = 3;
hill_k0   = 1.5;

hill_act = @(u, k, n)  u.^n ./ (k^n + u.^n);   % activation form (opposite of repression)

%% --- GROUND-TRUTH PARAMETERS & SYNTHETIC DATA ------------------------
% Used only to simulate synthetic data and to print a side-by-side
% comparison against what RI-SINDy recovers.
alpha_z = 8.0;
kxy     = 1.2;
kx      = 0.4;
ky_z    = 1.8;
ky      = 0.3;
kz      = 0.8;

t_end     = (N + 200) * dt;
t_span    = 0:dt:t_end;
signal_fn = @(t) 3.5 * exp(-0.15*t) .* (t > 0);   % decaying pulse (mimics IR/TNF stimulus)

ode_rhs = @(t, s) [ ...
    signal_fn(t)  -  kxy*s(1)*s(2)  -  kx*s(1); ...
    ky_z*s(3)     -  kxy*s(1)*s(2)  -  ky*s(2); ...
    alpha_z * hill_act(s(1), hill_k0, hill_n)  -  kz*s(3) ];

opts_ode   = odeset('RelTol',1e-9,'AbsTol',1e-11);
[t_ode, S] = ode45(ode_rhs, t_span, [0.1; 0.05; 0.02], opts_ode);

x_data_all = S(:,1);
y_data_all = S(:,2);
z_data_all = S(:,3);
t_all      = t_ode;
signal_all = signal_fn(t_all);

noise_level = 0.1;
rng(42);
x_data_all = x_data_all + noise_level*std(x_data_all)*randn(size(x_data_all));
y_data_all = y_data_all + noise_level*std(y_data_all)*randn(size(y_data_all));
z_data_all = z_data_all + noise_level*std(z_data_all)*randn(size(z_data_all));

x_data = x_data_all(1:N);
y_data = y_data_all(1:N);
z_data = z_data_all(1:N);
t      = t_all(1:N);
signal_train = signal_all(1:N);

fprintf('NF-kB: %d training samples, dt=%.4f\n\n', N, dt);

%% --- DERIVATIVES (computed on the FULL series, then truncated -- same
%      boundary-effect reasoning as Hes1) ---
dxdt_all = smooth_derivative(x_data_all, dt, 3, 11);
dydt_all = smooth_derivative(y_data_all, dt, 3, 11);
dzdt_all = smooth_derivative(z_data_all, dt, 3, 11);
dXdt_tr  = [dxdt_all(1:N), dydt_all(1:N), dzdt_all(1:N)];

%% --- LIBRARY (unified, 13 columns -- see STRUCTURAL NOTE above) -----------
col_names = {'1','S','x','x^2','x^3','y','y^2','y^3','z','z^2','z^3','x*y','Hill(x)'};

HillAct_x = hill_act(x_data, hill_k0, hill_n);
Theta_tr = [ ones(N,1), signal_train, ...
             x_data, x_data.^2, x_data.^3, ...
             y_data, y_data.^2, y_data.^3, ...
             z_data, z_data.^2, z_data.^3, ...
             x_data.*y_data, HillAct_x ];

%% --- CONSTRAINTS ------------------------------------------------------
% Column order: [1, S, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, x*y]
% (Hill(x) is column 13, pinned -- never enters these bounds).
% Each equation's inapplicable block (e.g. z-terms for the x/y
% equations) is forced to exactly zero via lb=ub=0, reproducing the
% original script's use of two separate, smaller library matrices.
lb{1} = [0, 0, -inf, -inf, -inf, -inf, -inf, -inf,   0,   0,   0, -inf];   % x equation
ub{1} = [0, inf,   0,    0,    0,    0,    0,    0,   0,   0,   0,    0];
lb{2} = [0, 0,    0,    0,    0, -inf, -inf, -inf,   0,   0,   0, -inf];   % y equation
ub{2} = [inf, 0, inf,  inf,  inf,    0,    0,    0,   0,   0,   0,    0];
lb{3} = [0, 0, -inf, -inf, -inf,   0,   0,   0, -inf, -inf, -inf,    0];   % z equation
ub{3} = [inf, 0,   0,    0,    0,   0,   0,   0,    0,    0,    0,    0];

pinned_col = 13;   % Hill(x)

%% --- DRIFT-BALANCE FUNCTION -----------------------------------------------
drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
    drift_balance_nfkb(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                        x_data, z_data, hill_k0, hill_n);

%% --- FIT ------------------------------------------------------------------
opts = struct();
opts.eta_pin     = 0.6;
opts.eta_weight  = 0.7;    % matches default; explicit for clarity
opts.eta_drain   = 0;      % NO drain-coefficient smoothing -- this script
                            % never used it, same situation as Hes1
opts.n_iter      = 20;
opts.w_threshold = 2;      % much lower than Hes1 (100) or Goodwin (10)
opts.prior       = 'SpikeSlab';   % matches default; explicit for clarity
opts.prior_params = struct('v0', 1e-4, 'v1', 1.0);

[Xi, Xi_var, diagnostics] = risindy(Theta_tr, dXdt_tr, drift_fn, pinned_col, lb, ub, opts); %#ok<ASGLU>

%% --- RESULTS ----------------------------------------------------------
fprintf('==========================================\n');
fprintf('  IDENTIFIED EQUATIONS  (RI-SINDy NF-kB)\n');
fprintf('==========================================\n\n');
eq_str = {'dx/dt','dy/dt','dz/dt'};
for eq = 1:3
    fprintf('%s =\n', eq_str{eq});
    for j = 1:numel(col_names)
        if abs(Xi(j,eq)) > 1e-6
            fprintf('   %+.4f * %s\n', Xi(j,eq), col_names{j});
        end
    end
    fprintf('\n');
end
fprintf('--- Ground Truth ---\n');
fprintf('dx/dt = +[signal]  %+.4f * x*y  %+.4f * x\n',    -kxy, -kx);
fprintf('dy/dt = %+.4f * z  %+.4f * x*y  %+.4f * y\n',  ky_z, -kxy, -ky);
fprintf('dz/dt = %+.4f * Hill(x)  %+.4f * z\n',         alpha_z, -kz);
fprintf('(note: ky_z*z in the y equation is not identifiable from this\n');
fprintf(' library -- z was intentionally not included as a candidate in\n');
fprintf(' the y equation, matching the original script.)\n');

%% --- FORWARD INTEGRATION ------------------------------------------------
rhs = @(xp,yp,zp,sp) [1, sp, xp, xp^2, xp^3, yp, yp^2, yp^3, zp, zp^2, zp^3, xp*yp, hill_act(xp,hill_k0,hill_n)] * Xi;
[x_id, y_id, z_id] = forward_integrate_nfkb(rhs, t_all, signal_all, x_data_all(1), y_data_all(1), z_data_all(1), dt);

%% --- FIT METRICS ------------------------------------------------------
eval_start = round(0.1 * N);
eval_idx   = eval_start:numel(t_all);
rmse = @(a,b) sqrt(mean((a-b).^2));
r2   = @(a,b) 1 - sum((a-b).^2)/sum((a-mean(a)).^2);

fprintf('\n==========================================\n');
fprintf('  FIT METRICS (eval t > %.1f)\n', t_all(eval_start));
fprintf('==========================================\n');
fprintf('       RMSE       R^2\n');
fprintf('  x:  %.4f    %.4f\n', rmse(x_data_all(eval_idx),x_id(eval_idx)), r2(x_data_all(eval_idx),x_id(eval_idx)));
fprintf('  y:  %.4f    %.4f\n', rmse(y_data_all(eval_idx),y_id(eval_idx)), r2(y_data_all(eval_idx),y_id(eval_idx)));
fprintf('  z:  %.4f    %.4f\n', rmse(z_data_all(eval_idx),z_id(eval_idx)), r2(z_data_all(eval_idx),z_id(eval_idx)));

%% --- UQ BAND VIA MONTE CARLO ------------------------------------------
% Xi_var is already in physical units (risindy.m denormalizes it
% internally) -- no manual rescaling needed, unlike the original script.
draw_fn = @(Xi_s) forward_integrate_nfkb( ...
    @(xp,yp,zp,sp) [1, sp, xp, xp^2, xp^3, yp, yp^2, yp^3, zp, zp^2, zp^3, xp*yp, hill_act(xp,hill_k0,hill_n)] * Xi_s, ...
    t_all, signal_all, x_data_all(1), y_data_all(1), z_data_all(1), dt);
addpath(genpath(fullfile(thisDir, '..', 'comparisons', 'utils')));
[Xenv_lo, Xenv_hi] = mc_uq_propagate(draw_fn, Xi, Xi_var, 500);

%% --- PLOTS: time-series fit with UQ bands ---------------------------------
figure('Color','w');
state_data    = {x_data_all, y_data_all, z_data_all};
state_id      = {x_id, y_id, z_id};
state_lo      = {Xenv_lo(:,1), Xenv_lo(:,2), Xenv_lo(:,3)};
state_hi      = {Xenv_hi(:,1), Xenv_hi(:,2), Xenv_hi(:,3)};
state_names   = {'x  (nuclear NF-\kappaB)', 'y  (I\kappaB\alpha protein)', 'z  (I\kappaB\alpha mRNA)'};
state_colors  = {[0.08 0.45 0.85], [0.85 0.20 0.10], [0.10 0.65 0.30]};

for s = 1:3
    subplot(3,1,s); hold on;
    fill([t_all; flipud(t_all)], [state_hi{s}; flipud(state_lo{s})], ...
        state_colors{s}, 'FaceAlpha',0.18, 'EdgeColor','none');
    plot(t_all, state_data{s}, '--', 'Color', state_colors{s}*0.6, 'LineWidth',1.0);
    plot(t_all, state_id{s},   '-',  'Color', state_colors{s},     'LineWidth',1.5);
    xline(N*dt, 'k:', 'LineWidth',1.2);
    ylabel(state_names{s}, 'FontSize',9);
    if s == 1
        title('RI-SINDy Identification: NF-\kappaB Network', 'FontSize',10);
        legend({'90% UQ','Synthetic data','Identified','Train | Extrap'}, ...
               'FontSize',8,'Location','northeast');
    end
    if s == 3, xlabel('Time', 'FontSize',9); end
    box on; set(gca,'Color','w','TickDir','out','FontSize',8);
end

%% --- PLOT: phase portrait -----------------------------------------------
figure('Color','w');
plot3(x_data_all, y_data_all, z_data_all, 'b-', 'LineWidth',0.8); hold on;
plot3(x_id, y_id, z_id, 'r--', 'LineWidth',1.5);
xlabel('x (NF-\kappaB)'); ylabel('y (I\kappaB)'); zlabel('z (mRNA)');
title('Phase Portrait: NF-\kappaB'); legend('Synthetic data','RI-SINDy model','Location','northeast');
grid on; view(35,25);


%% =========================================================================
%  FORWARD INTEGRATION (plain Euler step, time-varying signal S(t) --
%  kept local, same pattern as Hes1/Goodwin's local forward-integration
%  helpers)
%% =========================================================================
function [x_id, y_id, z_id] = forward_integrate_nfkb(rhsfun, t_all, signal_all, x0, y0, z0, dt)
    n_all = numel(t_all);
    x_id = zeros(n_all,1); x_id(1) = x0;
    y_id = zeros(n_all,1); y_id(1) = y0;
    z_id = zeros(n_all,1); z_id(1) = z0;
    for k = 2:n_all
        xp = x_id(k-1); yp = y_id(k-1); zp = z_id(k-1);
        sp = signal_all(k-1);
        dxyz = rhsfun(xp, yp, zp, sp);
        x_id(k) = xp + dt*dxyz(1);
        y_id(k) = yp + dt*dxyz(2);
        z_id(k) = zp + dt*dxyz(3);
    end
end