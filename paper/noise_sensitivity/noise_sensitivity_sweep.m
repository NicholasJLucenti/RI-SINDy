%% =========================================================================
%  Noise Sensitivity Sweep (supplementary, not in the paper)
%
%  Actually RUNS Traditional SINDy, E-SINDy, and RI-SINDy on the Goodwin
%  system at four noise levels (0%, 5%, 10%, 15%) and computes the three
%  metrics shown in noise_sensitivity_heatmap.m from scratch: correct
%  terms recovered, spurious terms identified, and mean relative L2
%  trajectory error. This is what noise_sensitivity_heatmap.m's
%  hardcoded numbers should ultimately come FROM -- run this and copy
%  the printed matrices into that script's DATA section to update the
%  static figure, or just call plot_metric_heatmap.m directly on this
%  script's live output (already done below).
%
%  GROUND-TRUTH TERM COUNT FLAG: the paper text and the existing
%  noise_sensitivity_heatmap.m both say 7 correct terms at 0% noise.
%  The Goodwin model implemented throughout this repo (see
%  paper/goodwin/goodwin_risindy.m) has only 6 nonzero ground-truth
%  terms: x and Hrep(z) in the x-equation, x and Hdeg(y) in the
%  y-equation, y and z in the z-equation. gt_mask below reflects that
%  6-term model. If the noise-sensitivity test in the paper used a
%  different/extended Goodwin variant with a 7th nonzero term, gt_mask
%  needs to be edited to match before these numbers can be compared
%  directly against the paper's figure.
%
%  REPRODUCIBILITY NOTE: E-SINDy involves bootstrap resampling, so its
%  exact numbers will vary slightly run to run even with a fixed seed
%  for the overall sweep, if n_boot or incl_threshold are changed.
%  RI-SINDy is refit fresh at every noise level here (NOT the hardcoded
%  paper coefficients) -- this is a live test of the same numerical
%  stability question discussed when tuning force_multiplier/etas/
%  thresholds by hand; if RI-SINDy's fit quality degrades sharply at a
%  noise level here, that's the same instability, just being measured
%  systematically instead of noticed by hand.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin not needed here)
%            Signal Processing Toolbox (sgolayfilt)
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));
addpath(genpath(fullfile(thisDir, '..', 'goodwin')));
addpath(genpath(fullfile(thisDir, '..', 'comparisons', 'utils')));
addpath(genpath(fullfile(thisDir, 'utils')));

%% --- SWEEP CONFIGURATION --------------------------------------------------
noise_levels   = [0, 0.05, 0.10, 0.15];
noiseLabels    = {'0%','5%','10%','15%'};
methodLabels   = {'SINDy','E-SINDy','RI-SINDy'};
n_boot_esindy  = 100;
incl_threshold = 0.5;
term_threshold = 1e-3;   % magnitude below which a coefficient counts as "not identified"
rng_seed_base  = 100;    % each noise level gets rng_seed_base + noise_index, for reproducibility

col_names = {'1','x','x^2','x^3','y','y^2','y^3','z','z^2','z^3','Hrep(z)','Hdeg(y)'};

% Ground-truth sparsity mask -- see GROUND-TRUTH TERM COUNT FLAG above.
gt_mask = false(12, 3);
gt_mask(2,1)  = true;  gt_mask(11,1) = true;   % x, Hrep(z)  -> x equation
gt_mask(2,2)  = true;  gt_mask(12,2) = true;   % x, Hdeg(y)  -> y equation
gt_mask(5,3)  = true;  gt_mask(8,3)  = true;   % y, z        -> z equation
n_correct_total = nnz(gt_mask);   % 6 -- see flag above if this should be 7

%% --- RESULT STORAGE ---------------------------------------------------
correctTerms  = NaN(numel(noise_levels), 3);
spuriousTerms = NaN(numel(noise_levels), 3);
relL2         = NaN(numel(noise_levels), 3);

%% --- SWEEP --------------------------------------------------------------
for ni = 1:numel(noise_levels)
    noise = noise_levels(ni);
    rng(rng_seed_base + ni);

    fprintf('=== Noise level %s ===\n', noiseLabels{ni});

    [t, Xtrue, Xnoisy, p, N_train] = generate_goodwin_data_sweep(noise);
    fitIdx = 1:N_train;
    dt = t(2) - t(1);
    Xfit = Xnoisy(fitIdx, :);
    dXdt_fit = derivative_sgolay_sweep(Xfit, dt);
    libFun = @(X, tt) lib_goodwin_sweep(X, tt, p.K0, p.n, p.Km);
    X0 = Xtrue(1, :);

    %% --- Traditional SINDy ---
    Xi_trad = run_traditional_sindy_3d(libFun, Xfit, t(fitIdx), dXdt_fit);
    rhs_trad = @(tt, X) (libFun(X', tt) * Xi_trad)';
    Xhat_trad = integrate_model(rhs_trad, t, X0);
    [correctTerms(ni,1), spuriousTerms(ni,1), relL2(ni,1)] = ...
        score_method(Xi_trad, gt_mask, term_threshold, Xtrue, Xhat_trad);
    fprintf('  Traditional SINDy: correct=%d spurious=%d relL2=%.3f\n', ...
            correctTerms(ni,1), spuriousTerms(ni,1), relL2(ni,1));

    %% --- E-SINDy ---
    Xi_ens = run_ensemble_sindy_3d(libFun, Xfit, t(fitIdx), dXdt_fit, ...
                                    n_boot_esindy, incl_threshold, 0.1);
    rhs_ens = @(tt, X) (libFun(X', tt) * Xi_ens)';
    Xhat_ens = integrate_model(rhs_ens, t, X0);
    [correctTerms(ni,2), spuriousTerms(ni,2), relL2(ni,2)] = ...
        score_method(Xi_ens, gt_mask, term_threshold, Xtrue, Xhat_ens);
    fprintf('  E-SINDy:            correct=%d spurious=%d relL2=%.3f\n', ...
            correctTerms(ni,2), spuriousTerms(ni,2), relL2(ni,2));

    %% --- RI-SINDy (refit fresh at this noise level -- NOT hardcoded) ---
    x_data = Xfit(:,1); y_data = Xfit(:,2); z_data = Xfit(:,3);
    Theta_tr = [build_poly_library(x_data, y_data, z_data, 3), ...
                (p.K0^p.n) ./ (p.K0^p.n + z_data.^p.n), ...
                y_data ./ (p.Km + y_data)];

    lb{1} = [ 0, -inf, -inf, -inf,    0,    0,    0,    0,    0,    0];
    ub{1} = [ 0,    0,    0,    0,    0,    0,    0,    0,    0,    0];
    lb{2} = [ 0,    2,    0,    0,    0,    0,    0,    0,    0,    0];
    ub{2} = [ 0,  inf,    0,    0,    0,    0,    0,    0,    0,    0];
    lb{3} = [ 0,    0,    0,    0,    0,    0,    0, -inf, -inf, -inf];
    ub{3} = [ 0,    0,    0,    0,  inf,  inf,  inf,    0,    0,    0];
    pinned_col = [11, 12];

    drift_fn = @(XiN_smooth, XiN_var, vi, col_scale, target_scale) ...
        drift_balance_goodwin(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                               x_data, y_data, z_data, p.K0, p.n, p.Km);

    opts = struct();
    opts.eta_pin      = 0.6;
    opts.eta_weight   = 0.7;
    opts.eta_drain    = 0.7;
    opts.n_iter       = 50;
    opts.w_threshold  = 10;
    opts.var_floor    = 1e-12;
    opts.prior        = 'Horseshoe';
    opts.prior_params = struct('tau0', 0.05, 'eps_sparsity', 1e-3);

    Xi_ri = risindy(Theta_tr, dXdt_fit, drift_fn, pinned_col, lb, ub, opts);
    Xi_ri(abs(Xi_ri) < 0.005) = 0;   % same post-hoc cleanup as goodwin_risindy.m

    rhs_ri = @(tt, X) (libFun(X', tt) * Xi_ri)';
    Xhat_ri = integrate_model(rhs_ri, t, X0);
    [correctTerms(ni,3), spuriousTerms(ni,3), relL2(ni,3)] = ...
        score_method(Xi_ri, gt_mask, term_threshold, Xtrue, Xhat_ri);
    fprintf('  RI-SINDy:           correct=%d spurious=%d relL2=%.3f\n\n', ...
            correctTerms(ni,3), spuriousTerms(ni,3), relL2(ni,3));
end

%% --- RESULTS ------------------------------------------------------------
fprintf('==========================================\n');
fprintf('  RESULT MATRICES (copy into noise_sensitivity_heatmap.m if desired)\n');
fprintf('==========================================\n');
fprintf('correctTerms =\n');  disp(correctTerms);
fprintf('spuriousTerms =\n'); disp(spuriousTerms);
fprintf('relL2 =\n');         disp(relL2);

%% --- FIGURE (same layout as noise_sensitivity_heatmap.m, live data) -------
figure('Color','w', 'Position',[100 100 1250 380], 'Name','Goodwin Noise Sensitivity (live)');

subplot(1,3,1)
plot_metric_heatmap(correctTerms, noiseLabels, methodLabels, ...
    'Correct Terms Recovered', 'blue_white', n_correct_total, [0, n_correct_total]);

subplot(1,3,2)
plot_metric_heatmap(spuriousTerms, noiseLabels, methodLabels, ...
    'Spurious Terms Identified', 'orange_white', [], []);

subplot(1,3,3)
plot_metric_heatmap(relL2, noiseLabels, methodLabels, ...
    'Relative L_2 Error', 'orange_white', [], [0, 0.6]);

sgtitle('Goodwin Oscillator: Noise Sensitivity (live sweep)', 'FontSize', 11, 'FontWeight', 'normal');


%% =========================================================================
%  LOCAL HELPERS
%% =========================================================================
function [n_correct, n_spurious, rel_l2] = score_method(Xi, gt_mask, term_threshold, Xtrue, Xhat)
    [n_correct, n_spurious] = count_recovered_terms(Xi, gt_mask, term_threshold);
    if any(~isfinite(Xhat(:)))
        % Diverged model: blank ALL metrics for this cell, matching the
        % convention already used in noise_sensitivity_heatmap.m (a
        % diverged model's term counts aren't meaningfully comparable
        % either, not just its trajectory error).
        n_correct  = NaN;
        n_spurious = NaN;
        rel_l2     = NaN;
        return;
    end
    per_dim = zeros(1, size(Xtrue,2));
    for d = 1:size(Xtrue,2)
        per_dim(d) = norm(Xhat(:,d) - Xtrue(:,d)) / norm(Xtrue(:,d));
    end
    rel_l2 = mean(per_dim);
end

function [t, Xtrue, Xnoisy, p, N_train] = generate_goodwin_data_sweep(noise_level)
    p.alpha = 8.0;  p.dx = 0.6;  p.K0 = 2.0;  p.n = 3;
    p.ky    = 2.0;
    p.beta  = 5.0;  p.Km = 1.0;
    p.kz    = 1.5;  p.dz = 0.8;

    dt = 0.05;
    N_train = 300;
    n_total = N_train + 200;

    rhs = @(tt,X) goodwin_rhs_sweep(X, p);
    tspan = (0 : dt : n_total*dt)';
    X0 = [1.5; 0.5; 1.0];
    [t, Xtrue] = ode45(rhs, tspan, X0, odeset('RelTol',1e-9,'AbsTol',1e-11));

    sig = std(Xtrue, 0, 1);
    Xnoisy = Xtrue + noise_level .* sig .* randn(size(Xtrue));
end

function dX = goodwin_rhs_sweep(X, p)
    x = X(1); y = X(2); z = X(3);
    Hrep = (p.K0^p.n) / (p.K0^p.n + z^p.n);
    Hdeg = y / (p.Km + y);
    dx = p.alpha*Hrep - p.dx*x;
    dy = p.ky*x - p.beta*Hdeg;
    dz = p.kz*y - p.dz*z;
    dX = [dx; dy; dz];
end

function dX = derivative_sgolay_sweep(X, dt)
    [m,n] = size(X);
    dX = zeros(m,n);
    for i = 1:n
        dX(:,i) = smooth_derivative(X(:,i), dt, 3, 11);
    end
end

function Theta = lib_goodwin_sweep(X, tt, K0, n, Km)
    x = X(:,1); y = X(:,2); z = X(:,3);
    Hrep = (K0^n) ./ (K0^n + z.^n);
    Hdeg = y ./ (Km + y);
    Theta = [build_poly_library(x, y, z, 3), Hrep, Hdeg];
end