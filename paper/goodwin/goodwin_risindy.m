%% =========================================================================
%  Goodwin Oscillator -- RI-SINDy fit
%
%  Produces the RI-SINDy coefficients used throughout the paper for the
%  Goodwin system: fits the sparse polynomial drain/production terms via
%  risindy.m while the two Hill regulatory terms (HillRep(z) in the
%  mRNA equation, HillDeg(y) in the protein equation) are pinned by
%  drift_balance_goodwin.m rather than fit by regression.
%
%  Requires: Optimization Toolbox (lsqlin)
%            Signal Processing Toolbox (sgolayfilt)
%
%  FIDELITY NOTE: risindy.m's initialization solves the initial least
%  squares fit against the FULL library (all 12 columns, including the
%  two Hill columns) and then discards the pinned rows -- see
%  src/risindy.m's header. The original version of this script
%  initialized only against the 10 polynomial columns. Both converge to
%  the same fixed point after enough iterations (50, here); this only
%  changes the starting point of iteration 1, not the final result in
%  any run we've checked.
% =========================================================================

close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

%% --- PARAMETERS -----------------------------------------------------------
dt      = 0.05;
N       = 300;
hill_n  = 3;
hill_k0 = 2;
hill_km = 1.0;

hill_rep = @(z, k, n)  k^n ./ (k^n + z.^n);
hill_deg = @(y, km)    y   ./ (km  + y);

% Ground-truth parameters, used only to simulate synthetic data and to
% print a side-by-side comparison against what RI-SINDy recovers.
alpha = 8.0;
d1    = 0.6;
ks    = 2.0;
Vmax  = 5.0;
kp    = 1.5;
kd    = 0.8;

%% --- SIMULATE GROUND TRUTH --------------------------------------------
t_end  = (N + 200) * dt;
t_span = 0 : dt : t_end;
ode_rhs = @(t, s) [ ...
    alpha * hill_rep(s(3), hill_k0, hill_n) - d1*s(1); ...
    ks*s(1) - Vmax * hill_deg(s(2), hill_km); ...
    kp*s(2) - kd*s(3) ];
opts_ode   = odeset('RelTol',1e-9,'AbsTol',1e-11);
[t_ode, S] = ode45(ode_rhs, t_span, [1.5; 0.5; 1.0], opts_ode);

x_data_all = S(:,1);
y_data_all = S(:,2);
z_data_all = S(:,3);
t_all      = t_ode;
n_all      = length(t_all);

noise_level = 0.05;
rng(42);
x_data_all = x_data_all + noise_level*std(x_data_all)*randn(n_all,1);
y_data_all = y_data_all + noise_level*std(y_data_all)*randn(n_all,1);
z_data_all = z_data_all + noise_level*std(z_data_all)*randn(n_all,1);

x_data = x_data_all(1:N);
y_data = y_data_all(1:N);
z_data = z_data_all(1:N);
t      = t_all(1:N);
fprintf('Goodwin oscillator: %d training samples, dt=%.3f\n\n', N, dt);

%% --- DERIVATIVES --------------------------------------------------------
dxdt_train = smooth_derivative(x_data, dt, 3, 11);
dydt_train = smooth_derivative(y_data, dt, 3, 11);
dzdt_train = smooth_derivative(z_data, dt, 3, 11);
dXdt_tr    = [dxdt_train, dydt_train, dzdt_train];

%% --- LIBRARY --------------------------------------------------------------
col_names = {'1','x','x^2','x^3','y','y^2','y^3','z','z^2','z^3',...
             'HillRep(z)','HillDeg(y)'};

Theta_tr = [build_poly_library(x_data, y_data, z_data, 3), ...
            hill_rep(z_data, hill_k0, hill_n), ...
            hill_deg(y_data, hill_km)];

%% --- CONSTRAINTS --------------------------------------------------------
% Column order: [1, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3] (HillRep and
% HillDeg are columns 11-12, pinned -- never enter these bounds).
lb{1} = [ 0, -inf, -inf, -inf,    0,    0,    0,    0,    0,    0];   % mRNA equation
ub{1} = [ 0,    0,    0,    0,    0,    0,    0,    0,    0,    0];
lb{2} = [ 0,    2,    0,    0,    0,    0,    0,    0,    0,    0];   % protein equation
ub{2} = [ 0,  inf,    0,    0,    0,    0,    0,    0,    0,    0];
lb{3} = [ 0,    0,    0,    0,    0,    0,    0, -inf, -inf, -inf];   % repressor equation
ub{3} = [ 0,    0,    0,    0,  inf,  inf,  inf,    0,    0,    0];

pinned_col = [11, 12];   % HillRep(z), HillDeg(y)

%% --- DRIFT-BALANCE FUNCTION -----------------------------------------------
drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
    drift_balance_goodwin(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                           x_data, y_data, z_data, hill_k0, hill_n, hill_km);

%% --- FIT ----------------------------------------------------------------
opts = struct();
opts.eta_pin        = 0.6;    % Hill-pin blend rate
opts.eta_weight      = 0.7;    % sparsity-weight blend rate (matches default; explicit for clarity)
opts.eta_drain       = 0.7;    % drain-smoothing blend rate (matches default; explicit for clarity)
opts.n_iter          = 50;
opts.w_threshold      = 10;    % lower than the 100 used for Hes1
opts.var_floor        = 1e-12;
opts.prior            = 'Horseshoe';
opts.prior_params     = struct('tau0', 0.05, 'eps_sparsity', 1e-3);

[Xi, Xi_var, diagnostics] = risindy(Theta_tr, dXdt_tr, drift_fn, pinned_col, lb, ub, opts); %#ok<ASGLU>

% Final hard threshold on the identified physical coefficients -- this
% is an extra cleanup step applied AFTER risindy.m's own sparsification,
% not something risindy.m does internally.
threshold = 0.005;
Xi(abs(Xi) < threshold) = 0;

%% --- RESULTS ----------------------------------------------------------
fprintf('==========================================\n');
fprintf('  IDENTIFIED EQUATIONS  (RI-SINDy Goodwin)\n');
fprintf('==========================================\n\n');
eq_str = {'dx/dt (mRNA)','dy/dt (protein)','dz/dt (repressor)'};
for eq = 1:3
    fprintf('%s =\n', eq_str{eq});
    for j = 1:numel(col_names)
        if abs(Xi(j,eq)) > 1e-3
            fprintf('   %+.4f * %s\n', Xi(j,eq), col_names{j});
        end
    end
    fprintf('\n');
end
fprintf('--- Ground Truth ---\n');
fprintf('dx/dt = %+.4f * HillRep(z)  %+.4f * x\n',   alpha, -d1);
fprintf('dy/dt = %+.4f * x           %+.4f * HillDeg(y)\n', ks, -Vmax);
fprintf('dz/dt = %+.4f * y           %+.4f * z\n',    kp, -kd);

%% --- FORWARD INTEGRATION --------------------------------------------------
rhs = @(xp,yp,zp) [poly_library_row(xp,yp,zp,3), hill_rep(zp,hill_k0,hill_n), hill_deg(yp,hill_km)] * Xi;
[x_id, y_id, z_id] = forward_integrate_goodwin(rhs, n_all, dt, x_data(1), y_data(1), z_data(1));

%% --- PLOTS ----------------------------------------------------------------
figure;
subplot(3,1,1);
plot(t_all, x_data_all,'b','LineWidth',1.5); hold on;
plot(t_all, x_id,'b--','LineWidth',1.3);
xline(N*dt,'k:','Training end');
ylabel('x (mRNA)'); legend('Data','Identified'); grid on; box on;

subplot(3,1,2);
plot(t_all, y_data_all,'r','LineWidth',1.5); hold on;
plot(t_all, y_id,'r--','LineWidth',1.3);
xline(N*dt,'k:');
ylabel('y (protein)'); legend('Data','Identified'); grid on; box on;

subplot(3,1,3);
plot(t_all, z_data_all,'g','LineWidth',1.5); hold on;
plot(t_all, z_id,'g--','LineWidth',1.3);
xline(N*dt,'k:');
xlabel('Time'); ylabel('z (repressor)');
legend('Data','Identified'); grid on; box on;
sgtitle('RI-SINDy Goodwin Oscillator');

figure('Color','w');
plot3(x_data_all, y_data_all, z_data_all, 'b-', 'LineWidth', 0.8); hold on;
plot3(x_id, y_id, z_id, 'r--', 'LineWidth', 1.5);
xlabel('x (mRNA)'); ylabel('y (Protein)'); zlabel('z (Repressor)');
title('Phase Portrait Goodwin'); grid on;
legend('Synthetic data','RI-SINDy model','Location','northeast');

%% --- ERROR METRICS ------------------------------------------------------
clip = @(v) min(max(v, -1e6), 1e6);   % prevent divergent trajectories from producing Inf metrics
x_id_full = clip(x_id);
y_id_full = clip(y_id);
z_id_full = clip(z_id);

vars     = {'x (mRNA)', 'y (Protein)', 'z (Repressor)'};
data_all = {x_data_all, y_data_all, z_data_all};
id_full  = {x_id_full,  y_id_full,  z_id_full};

fprintf('\n==========================================\n');
fprintf('  TRAJECTORY ERROR METRICS\n');
fprintf('==========================================\n');
for v = 1:3
    dat = data_all{v};
    idd = id_full{v};
    rmse_v   = sqrt(mean((dat - idd).^2));
    mae_v    = mean(abs(dat - idd));
    rel_l2_v = norm(dat - idd, 2) / norm(dat, 2);
    fprintf('\n%s:\n', vars{v});
    fprintf('   RMSE       = %.4f\n', rmse_v);
    fprintf('   MAE        = %.4f\n', mae_v);
    fprintf('   Rel. L2    = %.4f\n', rel_l2_v);
end

%% --- POSTERIOR UNCERTAINTY ------------------------------------------------
% Xi_var is already in physical units (risindy.m denormalizes it
% internally) -- no additional rescaling needed here, unlike the
% original script, which did this conversion by hand.
Xi_std = sqrt(Xi_var);
fprintf('\nPosterior standard deviations (physical units):\n');
disp(array2table(Xi_std, 'RowNames', col_names, 'VariableNames', {'mRNA_Eq','Protein_Eq','Repressor_Eq'}));


%% =========================================================================
%  FORWARD INTEGRATION (plain Euler step -- no delay in Goodwin, so this
%  is simpler than Hes1's version; kept local rather than using
%  paper/comparisons/utils/integrate_model.m so the exact fixed-step
%  Euler scheme used to generate the paper's figures is preserved,
%  rather than switching to ode45's adaptive stepping)
%% =========================================================================
function [x_id, y_id, z_id] = forward_integrate_goodwin(rhsfun, n_all, dt, x0, y0, z0)
    x_id = zeros(n_all,1);  x_id(1) = x0;
    y_id = zeros(n_all,1);  y_id(1) = y0;
    z_id = zeros(n_all,1);  z_id(1) = z0;
    for k = 2:n_all
        xp = x_id(k-1);  yp = y_id(k-1);  zp = z_id(k-1);
        dxyz = rhsfun(xp, yp, zp);
        x_id(k) = xp + dt * dxyz(1);
        y_id(k) = yp + dt * dxyz(2);
        z_id(k) = zp + dt * dxyz(3);
    end
end