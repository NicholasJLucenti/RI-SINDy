%% =========================================================================
%  Hes1 Phase-Space Field Decomposition -- LIVE FIT (supplementary, not
%  in the paper)
%
%  Unlike hes1_phase_field_decomposition.m (which uses the paper's
%  already-published, hardcoded RI-SINDy coefficients), this script
%  ACTUALLY RUNS RI-SINDy fresh -- same fitting pipeline as
%  paper/hes1_app/hes1_risindy.m -- then uses whatever it identifies to:
%    1. plot the identified trajectory out to t = 12 hours, and
%    2. produce the same three-panel vector-field decomposition (Total /
%       Polynomial-drain-only / Regulatory-only) as the hardcoded
%       version, but from the live fit.
%
%  This exists specifically so a trajectory that looks wrong (e.g. not
%  showing the stable oscillation seen in earlier tests) can be traced
%  back to what the CURRENT fit actually produced, rather than comparing
%  against a hardcoded reference that may be out of date with recent
%  tuning changes.
%
%  IMPORTANT: this script starts with `clear;` deliberately. hes1_risindy.m
%  was recently found to be missing that -- if you ran Goodwin or NF-kB
%  in the same MATLAB session beforehand without clearing the workspace,
%  variables like lb/ub/opts/N/dt left over from that run can silently
%  contaminate a Hes1 fit (same variable names, different intended
%  values) without ever throwing an error. If your trajectory suddenly
%  stopped looking like earlier tests for no clear reason, this is the
%  first thing to rule out -- run this script in a FRESH MATLAB session
%  (or after `clear all`) before concluding anything else is wrong.
%
%  APPROXIMATION NOTE (same as hes1_phase_field_decomposition.m): the
%  vector-field panels evaluate the Hill term using the INSTANTANEOUS y
%  at each grid point as a stand-in for the true delayed y_tau, since
%  there's no trajectory history at an arbitrary off-path grid location.
%  The identified TRAJECTORY itself (panel overlay and the separate
%  time-series plot) uses the real delay-aware integration, so only the
%  background field arrows are approximate, not the trajectory.
%
%  Requires: Optimization Toolbox (lsqlin)
%            Signal Processing Toolbox (sgolayfilt)
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));
addpath(genpath(fullfile(thisDir, '..', 'hes1_app')));

%% --- HYPERPARAMETERS (matching hes1_risindy.m) -------------------------
polyorder = 2;
dt        = 0.05;
tau       = 0.25;
N         = 290;
hill_n    = 9;
hill_k0   = 2.7;
hill_func = @(p,k,n) 1 ./ (1 + (p./k).^n);

T_END = 12;   % how far to plot/integrate the identified trajectory

%% --- LOAD DATA ----------------------------------------------------------
dataDir = fullfile(thisDir, '..', '..', 'Hes1 Data');
x_data   = readmatrix(fullfile(dataDir, 'interpmRNAData.csv'));
y_data   = readmatrix(fullfile(dataDir, 'interpHes1Data.csv'));
mRNA     = readmatrix(fullfile(dataDir, 'mRNA.csv'));
mRNAtime = readmatrix(fullfile(dataDir, 'mRNAtime.csv'));
hes1     = readmatrix(fullfile(dataDir, 'hes1.csv'));
hes1time = readmatrix(fullfile(dataDir, 'hes1time.csv'));

t = linspace(0, 15, numel(x_data))';

%% --- DELAYED STATE AND LIBRARY --------------------------------------------
y_tau     = get_delayed(t, y_data, tau);
HillDelay = hill_func(y_tau, hill_k0, hill_n);
Theta     = [build_poly_library(x_data, y_data, polyorder), HillDelay];

%% --- DERIVATIVES (full series, then truncate -- see hes1_risindy.m) -----
dxdt = smooth_derivative(x_data, dt, 3, 11);
dydt = smooth_derivative(y_data, dt, 3, 11);
dXdt_tr  = [dxdt(1:N), dydt(1:N)];
Theta_tr = Theta(1:N, :);

fprintf('Hes1 live phase-field fit: N = %d training points (of %d available)\n', N, numel(x_data));

%% --- CONSTRAINTS (matching hes1_risindy.m exactly) ------------------------
lb{1} = [-inf, -inf, -inf,    0,    0];
ub{1} = [ inf, -1.5,     0,    0,    0];
lb{2} = [   0,    0,     0, -inf, -inf];
ub{2} = [ inf,  inf,   inf,    0,    0];
pinned_col = 6;

%% --- DRIFT-BALANCE FUNCTION AND FIT ----------------------------------------
drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
    drift_balance_hes1(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                        x_data(1:N), y_data(1:N), hill_k0, hill_n);

opts = struct();
opts.eta_pin     = 0.4;
opts.eta_weight  = 0.4;
opts.eta_drain   = 0;      % this system never used drain smoothing -- see hes1_risindy.m
opts.n_iter      = 20;
opts.w_threshold = 1e2;
opts.prior       = 'SpikeSlab';

Xi = risindy(Theta_tr, dXdt_tr, drift_fn, pinned_col, lb, ub, opts);

col_names = {'1','M','M^2','P','P^2','HillDelay'};
fprintf('\nIdentified coefficients (this run):\n');
disp(table(Xi(:,1), Xi(:,2), 'RowNames', col_names, 'VariableNames', {'mRNA_Eq','Protein_Eq'}));

%% --- FORWARD INTEGRATION OUT TO T_END --------------------------------------
t_plot = (0:dt:T_END)';
rhs = @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,hill_k0,hill_n)] * Xi;
Xhat = forward_integrate_hes1_live(rhs, t, t_plot, x_data(1), y_data(1), y_data, tau, dt);
x_id = Xhat(:,1); y_id = Xhat(:,2);

if any(~isfinite(Xhat(:)))
    fprintf('\n*** WARNING: identified trajectory diverged before t = %d hours. ***\n', T_END);
    fprintf('*** This is a real result of the current fit, not a plotting error. ***\n\n');
end

%% --- FIGURE 1: time-series trajectory out to T_END -------------------------
figure('Color', 'w', 'Name', 'Hes1 live fit: identified trajectory');
subplot(2,1,1); hold on;
plot(t(t<=T_END), x_data(t<=T_END), 'k-', 'LineWidth', 1);
plot(mRNAtime(mRNAtime<=T_END), mRNA(mRNAtime<=T_END), '.', 'Color', [0.4 0.4 0.4], 'MarkerSize', 8);
if ~any(~isfinite(x_id))
    plot(t_plot, x_id, 'b--', 'LineWidth', 1.5);
end
ylabel('mRNA'); title(sprintf('RI-SINDy live fit, t = 0 to %d hrs', T_END));
legend({'Interpolated data','Observed data','RI-SINDy identified'}, 'Location','best');
box on; set(gca,'Color','w','TickDir','out');

subplot(2,1,2); hold on;
plot(t(t<=T_END), y_data(t<=T_END), 'k-', 'LineWidth', 1);
plot(hes1time(hes1time<=T_END), hes1(hes1time<=T_END), '.', 'Color', [0.4 0.4 0.4], 'MarkerSize', 8);
if ~any(~isfinite(y_id))
    plot(t_plot, y_id, 'r--', 'LineWidth', 1.5);
end
xlabel('Time (h)'); ylabel('Protein');
box on; set(gca,'Color','w','TickDir','out');

%% --- FIGURES 2-4: phase-field decomposition, trajectory clipped to T_END ---
grid_res = 20;
x_range = linspace(min(x_data), max(x_data), grid_res);
y_range = linspace(min(y_data), max(y_data), grid_res);
[Xg, Yg] = meshgrid(x_range, y_range);

field_types = {'Total Identified Field', 'Polynomial Field (Drain)', 'Regulatory Field (Drift-Balance)'};

for f = 1:3
    figure('Color', 'w', 'Name', sprintf('%s (live fit)', field_types{f}));
    hold on; grid on;

    if ~any(~isfinite(Xhat(:)))
        plot(x_id, y_id, 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('Identified trajectory (t<=%d)', T_END));
    end

    U = zeros(size(Xg));
    V = zeros(size(Yg));
    for i = 1:numel(Xg)
        xp = Xg(i); yp = Yg(i);
        H_inst    = hill_func(yp, hill_k0, hill_n);   % see APPROXIMATION NOTE above
        poly_part = poly_library_row(xp, yp, polyorder);
        phi_total = [poly_part, H_inst];

        switch f
            case 1
                U(i) = phi_total * Xi(:,1);
                V(i) = phi_total * Xi(:,2);
            case 2
                U(i) = poly_part * Xi(1:5,1);
                V(i) = poly_part * Xi(1:5,2);
            case 3
                U(i) = H_inst * Xi(6,1);
                V(i) = H_inst * Xi(6,2);
        end
    end

    L = sqrt(U.^2 + V.^2);
    L(L == 0) = 1;
    quiver(Xg, Yg, U./L, V./L, 0.5, 'Color', [0.55 0.55 0.55], 'AutoScale', 'off', 'DisplayName', field_types{f});

    plot(x_data(1), y_data(1), 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Start');

    xlabel('Hes1 mRNA'); ylabel('Hes1 Protein');
    title(sprintf('%s (live fit, t<=%d hrs)', field_types{f}, T_END));
    axis tight; box on;
    legend('Location', 'northeast');
end


%% =========================================================================
%  FORWARD INTEGRATION (delay-aware Euler step, matching hes1_risindy.m's
%  own forward_integrate_hes1 -- kept local rather than shared since each
%  file's use case differs slightly. Also stops early and fills the rest
%  with NaN if the trajectory blows up, so a diverging fit shows up
%  immediately as a truncated/NaN trajectory rather than a silent
%  runaway plot.)
%% =========================================================================
function Xhat = forward_integrate_hes1_live(rhsfun, t, t_plot, x0, y0, y_data, tau, dt)
    x_id = zeros(numel(t_plot),1); x_id(1) = x0;
    y_id = zeros(numel(t_plot),1); y_id(1) = y0;
    for k = 2:numel(t_plot)
        xp = x_id(k-1); yp = y_id(k-1);
        if k <= round(tau/dt)+1
            y_tau_k = interp1(t, y_data(1:numel(t)), max(t_plot(k)-tau,0), 'linear');
        else
            y_tau_k = y_id(k - round(tau/dt));
        end
        dxy = rhsfun(xp, yp, y_tau_k);
        x_id(k) = xp + dt*dxy(1);
        y_id(k) = yp + dt*dxy(2);
        if ~all(isfinite([x_id(k), y_id(k)])) || max(abs([x_id(k), y_id(k)])) > 1e6
            x_id(k:end) = NaN; y_id(k:end) = NaN;
            break;
        end
    end
    Xhat = [x_id, y_id];
end