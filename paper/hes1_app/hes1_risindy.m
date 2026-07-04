%% =========================================================================
%  Hes1 mRNA-Protein Network -- RI-SINDy fit
%
%  Produces the RI-SINDy coefficients used throughout the paper for the
%  Hes1 system: fits the sparse polynomial drain terms via risindy.m
%  while the delayed Hill production term (mRNA equation only) is pinned
%  by drift_balance_hes1.m rather than fit by regression.
%
%  Requires: Optimization Toolbox (lsqlin)
%            Signal Processing Toolbox (sgolayfilt)
%
%  SCOPE NOTE: forward integration and the UQ band here only cover the
%  available interpolated window (t = 0-15 hrs). An earlier version of
%  this script additionally validated out to t = 35 hrs against an
%  extrapolated data file (FH3mRNA_EXTRAP.mat / FH3hes1_EXTRAP.mat) that
%  is not in this repository -- add it to "Hes1 Data" and extend T_END
%  below to restore that.
%
%  OMITTED FROM THIS VERSION: the Fourier-residual correction step and
%  the (already-commented-out) phase-plane field decomposition figure
%  that appeared in earlier versions of this script. The Fourier
%  residual material now lives in the paper's Section 3.1 rather than
%  the core fitting pipeline -- say the word if you want it as its own
%  paper/hes1/hes1_fourier_residual.m script. Figure output here is also
%  simplified to one clean UQ trajectory plot rather than the four
%  near-duplicate black-background figures the original produced.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));
addpath(genpath(fullfile(thisDir, '..', 'comparisons', 'utils')));

%% --- HYPERPARAMETERS ------------------------------------------------------
polyorder = 2;
dt        = 0.05;
tau       = 0.25;
N         = 290;          % training window (of 301 available points)
hill_n    = 9;
hill_k0   = 2.7;
hill_func = @(p,k,n) 1 ./ (1 + (p./k).^n);

%% --- LOAD DATA --------------------------------------------------------
dataDir = fullfile(thisDir, '..', '..', 'Hes1 Data');

x_data   = readmatrix(fullfile(dataDir, 'interpmRNAData.csv'));
y_data   = readmatrix(fullfile(dataDir, 'interpHes1Data.csv'));
mRNA     = readmatrix(fullfile(dataDir, 'mRNA.csv'));
mRNAtime = readmatrix(fullfile(dataDir, 'mRNAtime.csv'));
hes1     = readmatrix(fullfile(dataDir, 'hes1.csv'));
hes1time = readmatrix(fullfile(dataDir, 'hes1time.csv'));

t     = linspace(0, 15, numel(x_data))';
t_all = t;   % see SCOPE NOTE above

%% --- DELAYED STATE AND LIBRARY --------------------------------------------
y_tau     = get_delayed(t, y_data, tau);
HillDelay = hill_func(y_tau, hill_k0, hill_n);
Theta     = [build_poly_library(x_data, y_data, polyorder), HillDelay];

%% --- NUMERICAL DERIVATIVES --------------------------------------------
% Computed on the FULL series, THEN truncated to the training window --
% matching the original process exactly, since sgolayfilt's boundary
% behavior differs if computed on an already-truncated series.
dxdt = smooth_derivative(x_data, dt, 3, 11);
dydt = smooth_derivative(y_data, dt, 3, 11);

dXdt_tr  = [dxdt(1:N), dydt(1:N)];
Theta_tr = Theta(1:N, :);

fprintf('Hes1 RI-SINDy: N = %d training points (of %d available), dt = %.3f\n', ...
        N, numel(x_data), dt);

%% --- CONSTRAINTS --------------------------------------------------------
% Column order: [1, x, x^2, y, y^2] (HillDelay is column 6, pinned --
% never enters these bounds; risindy.m never passes it to lsqlin).
lb{1} = [-inf, -inf, -inf,    0,    0];   % mRNA equation
ub{1} = [ inf, -1.5,     0,    0,    0];
lb{2} = [   0,    0,     0, -inf, -inf];   % protein equation
ub{2} = [ inf,  inf,   inf,    0,    0];

pinned_col = 6;   % HillDelay is the last column

%% --- DRIFT-BALANCE FUNCTION -----------------------------------------------
drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
    drift_balance_hes1(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                        x_data(1:N), y_data(1:N), hill_k0, hill_n);

%% --- FIT ----------------------------------------------------------------
opts = struct();
opts.eta_pin     = 0.4;   % matches the original script's single eta, used
opts.eta_weight  = 0.4;   % identically for both the pin blend and the
                           % sparsity-weight blend
opts.eta_drain   = 0;     % NO drain-coefficient smoothing -- this script
                           % never used it; eta_drain=0 makes XiN_smooth
                           % track XiN exactly every iteration, reproducing
                           % that behavior exactly under the generalized core
opts.n_iter      = 20;
opts.w_threshold = 1e2;
opts.prior       = 'SpikeSlab';

[Xi, Xi_var, diagnostics] = risindy(Theta_tr, dXdt_tr, drift_fn, pinned_col, lb, ub, opts); %#ok<ASGLU>

%% --- RESULTS TABLE --------------------------------------------------------
col_names = {'1','M','M^2','P','P^2','HillDelay'};
fprintf('\nIdentified coefficients:\n');
disp(table(Xi(:,1), Xi(:,2), 'RowNames', col_names, 'VariableNames', {'mRNA_Eq','Protein_Eq'}));

fprintf('Identified equations (physical units):\n');
eq_names = {'dM/dt','dP/dt'};
for v = 1:2
    terms = '';
    for c = 1:numel(col_names)
        if abs(Xi(c,v)) > 1e-3
            if isempty(terms)
                terms = sprintf('%.5f*%s', Xi(c,v), col_names{c});
            else
                terms = sprintf('%s + %.5f*%s', terms, Xi(c,v), col_names{c});
            end
        end
    end
    fprintf('%s = %s\n', eq_names{v}, terms);
end

%% --- FORWARD INTEGRATION --------------------------------------------------
rhs  = @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,hill_k0,hill_n)] * Xi;
Xhat = forward_integrate_hes1(rhs, t, t_all, x_data(1), y_data(1), y_data, tau, dt);

%% --- UQ BAND VIA MONTE CARLO ------------------------------------------
fprintf('\nPropagating RI-SINDy UQ band (n=100 Monte Carlo draws)...\n');
draw_fn = @(Xi_s) forward_integrate_hes1( ...
    @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,hill_k0,hill_n)] * Xi_s, ...
    t, t_all, x_data(1), y_data(1), y_data, tau, dt);
[Xenv_lo, Xenv_hi] = mc_uq_propagate(draw_fn, Xi, Xi_var, 100);

%% --- FIGURE: trajectories with 90% UQ band -------------------------------
figure('Color', 'w');
ax1 = axes; hold on;
fill([0, t(N), t(N), 0], [-2, -2, 10, 10], [0.75 0.75 0.75], ...
    'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([t_all; flipud(t_all)], [Xenv_hi(:,1); flipud(Xenv_lo(:,1))], [0.5 0.5 1], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none');
fill([t_all; flipud(t_all)], [Xenv_hi(:,2); flipud(Xenv_lo(:,2))], [1 0.5 0.5], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none');

plot(t_all, Xhat(:,1), 'b--', 'LineWidth', 1.2);
plot(t_all, Xhat(:,2), 'r--', 'LineWidth', 1.2);
plot(t_all, x_data, 'b-', 'LineWidth', 1.2);
plot(t_all, y_data, 'r-', 'LineWidth', 1.2);
plot(mRNAtime, mRNA, 'o', 'Color', 0.7*[0.5 0.5 1], 'MarkerSize', 5, 'LineWidth', 1);
plot(hes1time, hes1, 'x', 'Color', 0.7*[1 0.5 0.5], 'MarkerSize', 5, 'LineWidth', 1);

xlim([0 10]); ylim([-1 10]);
xlabel('Time (h)', 'FontSize', 9);
ylabel('protein/mRNA concentration', 'FontSize', 9);
legend({'mRNA 90% UQ','Protein 90% UQ','RI-SINDy mRNA','RI-SINDy Protein', ...
        'Interpolated mRNA','Interpolated Protein','Observed mRNA','Observed Protein'}, ...
        'FontSize', 8, 'Location', 'northeast', 'Box','on');
set(ax1, 'Color','w','Box','on','TickDir','out','LineWidth',0.6,'FontSize',8, ...
         'XColor',[0.1 0.1 0.1],'YColor',[0.1 0.1 0.1],'XGrid','off','YGrid','off');
title('RI-SINDy Identified Trajectories: Hes1');


%% =========================================================================
%  FORWARD INTEGRATION (delay-aware Euler step; same pattern used in
%  paper/comparisons/hes1_comparisons.m -- kept local to each file for
%  now rather than promoted to a shared utility)
%% =========================================================================
function Xhat = forward_integrate_hes1(rhsfun, t, t_all, x0, y0, y_data, tau, dt)
    x_id = zeros(numel(t_all),1); x_id(1) = x0;
    y_id = zeros(numel(t_all),1); y_id(1) = y0;
    for k = 2:numel(t_all)
        xp = x_id(k-1); yp = y_id(k-1);
        if k <= round(tau/dt)+1
            y_tau_k = interp1(t, y_data(1:numel(t)), max(t_all(k)-tau,0), 'linear');
        else
            y_tau_k = y_id(k - round(tau/dt));
        end
        dxy = rhsfun(xp, yp, y_tau_k);
        x_id(k) = xp + dt*dxy(1);
        y_id(k) = yp + dt*dxy(2);
    end
    Xhat = [x_id, y_id];
end