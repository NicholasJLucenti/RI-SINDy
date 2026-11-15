%% =========================================================================
%  Hes1 mRNA-Protein Network -- RI-SINDy fit
%
%  Data: Hes1 mRNA and protein (Hirata et al. 2002), sum-of-sines
%  interpolant on 0-15 h ("Hes1 Data/"). Measurements end at about 12 h
%  (mRNA) and 8.3 h (protein); the interpolant beyond that is extrapolated.
%
%  Model class (M = mRNA, P = protein):
%      dM/dt = terms of [1, M, M^2, P, P^2] + xi_R * Hill(P(t - tau))
%      dP/dt = terms of [1, M, M^2, P, P^2]
%  The delayed Hill repression of mRNA by protein is the regulatory term,
%  set by the drift balance (xi_R = -D_B/D_R, Eq. 25, with xi_R >= 0).
%
%  Same algorithm block, iteration settings, stopping rule and threshold as
%  the p53 scripts. Bounds are kinetic signs plus a literature half-life
%  for each decay term (BOUNDS block).
%
%  Output (console): Table 2 RI-SINDy column with 95% credible intervals,
%  convergence and stopping rule, sign of the unclipped xi_R, library
%  condition numbers, trajectory error (Table 3 row), behaviour of the
%  free-running model. Figures: (1) library correlation (Fig 4a);
%  (2) trajectories with 90% Monte Carlo band; (3) convergence trace.
%  Baselines: paper/comparisons/hes1_comparisons.m.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr).
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));
addpath(genpath(fullfile(thisDir, '..', 'comparisons', 'utils')));   % heatmap, Monte Carlo band
dataDir = fullfile(thisDir, '..', '..', 'Hes1 Data');

%% --- SETTINGS -------------------------------------------------------------
rng(42);                  % Monte Carlo UQ draws
dt          = 0.05;       % h
tau         = 0.25;       % h; delay of the protein -> mRNA repression
hill_k0     = 2.7;        % Hill parameters from the Hes1 literature [1]
hill_n      = 9;
t_half_mRNA = 24.1/60;    % h; Hes1 mRNA half-life (Hirata et al. 2002)
t_half_prot = 22.3/60;    % h; Hes1 protein half-life (Hirata et al. 2002)
train_frac  = 0.9;        % first 90% of the window is fitted, the rest held out
lambda      = 0.05;       % |xi| below this is treated as zero (every method)
T_sim       = 120;        % h; free run of the identified model
osc_frac    = 0.2;        % sustained = SD over the last half > osc_frac x data SD
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;            % stopping rule: converged if xi_R moves < 1% from K = 60 to 100

col_names = {'1','M','M^2','P','P^2','Hill'};
reg_col   = [6, 0];       % regulatory column per equation (0 = none)
dvar      = 2;            % the delayed variable is the protein
hill      = @(d) 1 ./ (1 + (max(d, 0)/hill_k0).^hill_n);
libfun    = @(m, p, d) [ones(size(m)), m, m.^2, p, p.^2, hill(d)];

%% --- DATA, LIBRARY, DERIVATIVES -------------------------------------------
M = readmatrix(fullfile(dataDir, 'interpmRNAData.csv'));
P = readmatrix(fullfile(dataDir, 'interpHes1Data.csv'));
obs = {readmatrix(fullfile(dataDir, 'mRNAtime.csv')), readmatrix(fullfile(dataDir, 'mRNA.csv')); ...
       readmatrix(fullfile(dataDir, 'hes1time.csv')), readmatrix(fullfile(dataDir, 'hes1.csv'))};
t = linspace(0, 15, numel(M))';  X = [M, P];
N = round(train_frac * numel(t));
dXdt  = [smooth_derivative(M, dt, 3, 11), smooth_derivative(P, dt, 3, 11)];  % full series, then truncated
Theta = libfun(M, P, get_delayed(t, P, tau));
Th_tr = Theta(1:N,:);  dX_tr = dXdt(1:N,:);
fprintf('Hes1 RI-SINDy: N = %d training points (of %d), dt = %.3f h\n', N, numel(t), dt);

%% --- BOUNDS on physical coefficients, columns [1, M, M^2, P, P^2] ---------
% mRNA:    no constant; every polynomial term <= 0 (the Hill term is the only
%          production); M decays at least as fast as the mRNA half-life.
% protein: no constant; M, M^2 >= 0 (translation); P, P^2 <= 0; P decays at
%          least as fast as the protein half-life.
lb = {[0, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf]};
ub = {[0, -log(2)/t_half_mRNA, 0, 0, 0], [0, inf, inf, -log(2)/t_half_prot, 0]};

%% --- FIT ------------------------------------------------------------------
tic;
[Xi, Xi_var, ok] = fit_risindy(Th_tr, dX_tr, lb, ub, reg_col, 'drift', ri_opts, K_check);
t_fit = toc;
xiR_raw = -mean(Th_tr(:,1:5) * Xi(1:5,1)) / mean(Th_tr(:,6));     % before the sign bound
Xi(abs(Xi) < lambda) = 0;
sd = sqrt(Xi_var) .* (Xi ~= 0);

%% --- RESULTS: Table 2, RI-SINDy column ------------------------------------
fprintf('\nStopping rule: K_max = %d; converged if xi_R changes by < 1%% between K = %d and %d: %s (%.2f s)\n', ...
        ri_opts.n_iter, ri_opts.n_iter, K_check, mat2str(ok), t_fit);
fprintf('Settings: tau = %.2f h, K0 = %.1f, n = %d, half-lives %.1f / %.1f min, lambda = %.2f, eta = %.1f, prior %s\n', ...
        tau, hill_k0, hill_n, 60*t_half_mRNA, 60*t_half_prot, lambda, ri_opts.eta_pin, ri_opts.prior);
fprintf('\nTable 2, RI-SINDy (coefficient [95%% credible interval], physical units)\n');
fprintf('%-6s %32s %32s\n', 'term', 'mRNA', 'protein');
for k = 1:6
    s = cell(1, 2);
    for v = 1:2
        if Xi(k,v) == 0, s{v} = '0';
        else, s{v} = sprintf('%+.4f [%+.3f, %+.3f]', Xi(k,v), Xi(k,v) - 1.96*sd(k,v), Xi(k,v) + 1.96*sd(k,v)); end
    end
    fprintf('%-6s %32s %32s\n', col_names{k}, s{:});
end
fprintf('\nSign of xi_R before the bound xi_R >= 0: %.3f (bound active: %d)\n', xiR_raw, xiR_raw < 0);
ThN = Th_tr ./ vecnorm(Th_tr);
fprintf('Condition number of the unit-norm library: without Hill %.1f, with Hill %.1f (ratio %.2f)\n', ...
        cond(ThN(:,1:5)), cond(ThN), cond(ThN)/cond(ThN(:,1:5)));
fprintf('  (without the constant column: %.1f and %.1f)\n', cond(ThN(:,2:5)), cond(ThN(:,2:6)));

%% --- TRAJECTORY ERROR (Table 3 row) AND BEHAVIOUR -------------------------
Xs = simulate(Xi, libfun, t, X, dvar, tau, dt, T_sim, [0 0]);
s  = score_model(Xs, X, dt, osc_frac);   sd0 = score_model(X, X, dt, osc_frac);
[mode_T, mode_g] = linear_mode(Xi, libfun, dvar, tau, mean(X));
ts = (0:size(Xs,1)-1)'*dt;
rmse_obs = zeros(1, 2);
for v = 1:2
    tk = obs{v,1} <= t(end);
    rmse_obs(v) = sqrt(mean((interp1(ts, Xs(:,v), obs{v,1}(tk)) - obs{v,2}(tk)).^2));
end
fprintf('\nTable 3, RI-SINDy (t = 0-%g h, against the interpolant)\n', t(end));
fprintf('  mRNA     RMSE = %.4f   Rel. L2 = %.4f   (RMSE vs measurements %.4f)\n', s.rmse(1), s.rel(1), rmse_obs(1));
fprintf('  Protein  RMSE = %.4f   Rel. L2 = %.4f   (RMSE vs measurements %.4f)\n', s.rmse(2), s.rel(2), rmse_obs(2));
fprintf('  System-wide rel. L2 (both variables stacked) = %.4f\n', s.rel_sys);
fprintf('\nFree-running model: mean level %+.0f%% / %+.0f%%, amplitude ratio %.2f / %.2f (mRNA / protein)\n', ...
        100*s.level(1), 100*s.level(2), s.amp(1), s.amp(2));
fprintf('  sustained = %d, period %.2f h (data %.2f h), protein lag %.2f h (data %.2f h)\n', ...
        s.sustained, s.period, sd0.period, s.lag, sd0.lag);
fprintf('  linearized mode: period %.2f h, amplitude x %.2f per cycle\n', mode_T, mode_g);

%% --- FIGURE (1): library correlation (Fig 4a) -----------------------------
plot_collinearity_heatmap(Th_tr(:,2:end), col_names(2:end), 'hes1');

%% --- FIGURE (2): trajectories with 90% Monte Carlo band -------------------
draw_fn = @(Xi_s) win(simulate(Xi_s, libfun, t, X, dvar, tau, dt, t(end), [0 0]), numel(t));
[lo, hi] = mc_uq_propagate(draw_fn, Xi, Xi_var .* (Xi ~= 0), 100);
figure('Color', 'w');  hold on;
fill([0, t(N), t(N), 0], [-2, -2, 10, 10], [0.75 0.75 0.75], 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
fill([t; flipud(t)], [hi(:,1); flipud(lo(:,1))], [0.5 0.5 1], 'FaceAlpha', 0.18, 'EdgeColor', 'none');
fill([t; flipud(t)], [hi(:,2); flipud(lo(:,2))], [1 0.5 0.5], 'FaceAlpha', 0.18, 'EdgeColor', 'none');
plot(t, Xs(1:numel(t),1), 'b--', t, Xs(1:numel(t),2), 'r--', 'LineWidth', 1.2);
plot(t, M, 'b-', t, P, 'r-', 'LineWidth', 1.2);
plot(obs{1,1}, obs{1,2}, 'o', 'Color', 0.7*[0.5 0.5 1], 'MarkerSize', 5);
plot(obs{2,1}, obs{2,2}, 'x', 'Color', 0.7*[1 0.5 0.5], 'MarkerSize', 5);
xlim([0 15]);  ylim([-1 10]);  box on;
xlabel('Time (h)');  ylabel('protein/mRNA concentration');
legend({'mRNA 90% UQ', 'Protein 90% UQ', 'RI-SINDy mRNA', 'RI-SINDy Protein', 'Interpolated mRNA', ...
        'Interpolated Protein', 'Observed mRNA', 'Observed Protein'}, 'FontSize', 8, 'Location', 'northeast');
title('RI-SINDy Identified Trajectories: Hes1 (shaded: training window)');

%% --- FIGURE (3): convergence of xi_R --------------------------------------
iters = [1:10, 12:3:30, 35:5:K_check];  tr = zeros(size(iters));  nact = tr;
for k = 1:numel(iters)
    o = ri_opts;  o.n_iter = iters(k);
    Xk = fit_risindy(Th_tr, dX_tr, lb, ub, reg_col, 'drift', o, 0);
    tr(k) = Xk(6,1);  nact(k) = nnz(abs(Xk) >= lambda);
end
figure('Color', 'w', 'Position', [100 100 760 330]);
subplot(1, 2, 1);
semilogy(iters, abs(tr - tr(end)) / abs(tr(end)) + 1e-8, 'b-', 'LineWidth', 1.2);  hold on;
xline(ri_opts.n_iter, 'k--');  yline(0.01, 'k:');  box on;
xlabel('iteration K');  ylabel('|\xi_R(K) - \xi_R(100)| / |\xi_R(100)|');
subplot(1, 2, 2);
plot(iters, nact, 'k-', 'LineWidth', 1.2);  xline(ri_opts.n_iter, 'k--');  box on;
xlabel('iteration K');  ylabel('active terms (|\xi| \geq \lambda)');
sgtitle('Hes1: convergence of RI-SINDy');

function Y = win(X, n)
    Y = X(1:n, :);
end


%% =========================================================================
%  ALGORITHM BLOCK (identical in every Hes1 and p53 script)
%% =========================================================================
function R = fit_all_methods(t, X, dXdt, libfun_kn, dvar, tau, k0, n, lb, ub, reg_col, lambda, ri_opts, K_check, sr3, use)
    % Fits the nine methods listed in the comparison scripts. lb{v}, ub{v}:
    % physical bounds on every column except the Hill column, which is the
    % last column and active in the equation with reg_col > 0.
    Theta = libfun_kn(X(:,1), X(:,2), get_delayed(t, X(:,dvar), tau), k0, n);
    p = size(Theta, 2);  nv = size(dXdt, 2);  h = p;  rv = find(reg_col > 0);
    lbH = lb;  ubH = ub;
    for v = 1:nv
        if reg_col(v) > 0, lbH{v} = [lb{v}, 0];  ubH{v} = [ub{v}, inf];
        else,              lbH{v} = [lb{v}, 0];  ubH{v} = [ub{v}, 0];   end
    end
    nm = 9;  R.Xi = repmat({zeros(p, nv)}, 1, nm);  R.ok = false(1, nm);  R.secs = zeros(1, nm);
    R.k0 = k0*ones(1, nm);  R.n = n*ones(1, nm);  R.off = zeros(nm, 2);
    keepv = @(v) h*(v == rv);
    for m = find(use)
        tic;
        switch m
            case {1, 2}                                   % RI-SINDy / LS-Hill ablation
                modes = {'drift', 'ls'};
                [Xi, ~, R.ok(m)] = fit_risindy(Theta, dXdt, lb, ub, reg_col, modes{m}, ri_opts, K_check);
            case 3                                        % bounded SINDy
                Xi = zeros(p, nv);
                for v = 1:nv, Xi(:,v) = stlsq_bounded(Theta, dXdt(:,v), lbH{v}, ubH{v}, lambda, nonzeros(keepv(v))); end
                R.ok(m) = true;
            case 4                                        % Hill removed from the library
                Xi = zeros(p, nv);
                for v = 1:nv, Xi(1:p-1,v) = stlsq_bounded(Theta(:,1:p-1), dXdt(:,v), lb{v}, ub{v}, lambda, []); end
                R.ok(m) = true;
            case 5                                        % Ensemble-SINDy
                N = size(Theta, 1);  nb = 100;  B = zeros(p, nv, nb);
                for b = 1:nb
                    idx = randi(N, N, 1);
                    for v = 1:nv, B(:,v,b) = stlsq_bounded(Theta(idx,:), dXdt(idx,v), lbH{v}, ubH{v}, lambda, nonzeros(keepv(v))); end
                end
                Xi = zeros(p, nv);
                for v = 1:nv
                    for j = 1:p
                        kept = squeeze(B(j,v,:));  kept = kept(kept ~= 0);
                        if numel(kept) >= 0.5*nb, Xi(j,v) = median(kept); end
                    end
                end
                R.ok(m) = true;
            case 6                                        % Nullcline-SINDy
                sc = @(off) nullcline_score(off, t, X, dXdt, libfun_kn, dvar, tau, k0, n, lbH, ubH, lambda, keepv);
                off = fminsearch(sc, [0 0], optimset('Display', 'off', 'MaxIter', 80));
                [~, Xi] = sc(off);  R.off(m,:) = off;  R.ok(m) = true;
            case {7, 8}                                   % SR3, (K0, n) searched / fixed
                [Xi, R.k0(m), R.n(m)] = sr3_fit(t, X, dXdt, libfun_kn, dvar, tau, lbH, ubH, rv, lambda, ...
                                                 sr3, m == 8, k0, n);
                R.ok(m) = true;
            case 9                                        % Traditional SINDy, no bounds
                Xi = zeros(p, nv);
                for v = 1:nv, Xi(:,v) = stlsq_bounded(Theta, dXdt(:,v), -inf(1,p), inf(1,p), lambda, []); end
                R.ok(m) = true;
        end
        Xi(abs(Xi) < lambda) = 0;
        R.Xi{m} = Xi;  R.secs(m) = toc;
    end
end

function [Xi, Xi_var, ok] = fit_risindy(Theta, dXdt, lb, ub, reg_col, mode, opts, K_check)
    % RI-SINDy through src/risindy.m. reg_col(v) is the regulatory column of
    % equation v (0 = none); those columns are never regressed. lb{v}, ub{v}
    % are PHYSICAL bounds on the other columns and are converted here to the
    % normalized units risindy.m works in. mode 'drift' = drift balance
    % (RI-SINDy); 'ls' = least squares on the residual (ablation).
    pinned = unique(reg_col(reg_col > 0));
    free   = setdiff(1:size(Theta, 2), pinned);
    cs = vecnorm(Theta(:, free), 2, 1);  ts = vecnorm(dXdt, 2, 1);
    lbN = cell(1, numel(lb));  ubN = lbN;
    for v = 1:numel(lb), lbN{v} = lb{v} .* cs / ts(v);  ubN{v} = ub{v} .* cs / ts(v); end
    f = @(XiN, XiN_var, vi, col_scale, target_scale) ...
        reg_coef(XiN, XiN_var, vi, col_scale, target_scale, Theta, dXdt, reg_col, pinned, free, mode);
    [Xi, Xi_var] = risindy(Theta, dXdt, f, pinned, lbN, ubN, opts);
    ok = true;
    if K_check > 0
        o = opts;  o.n_iter = K_check;
        Xi2 = risindy(Theta, dXdt, f, pinned, lbN, ubN, o);
        ok = all(abs(Xi2(pinned,:) - Xi(pinned,:)) <= 0.01*abs(Xi(pinned,:)) + 1e-6, 'all');
    end
end

function [pin_val, pin_var] = reg_coef(XiN, XiN_var, vi, cs, ts, Theta, dXdt, reg_col, pinned, free, mode)
    % Regulatory coefficient of equation vi, in normalized units.
    % 'drift': xi_R = -D_B/D_R (Eq. 25), variance by the delta method (Eq. 26-27).
    % 'ls':    xi_R = least-squares fit of the Hill column to the residual.
    % Both carry the sign bound xi_R >= 0 (the direction of regulation is known).
    pin_val = zeros(numel(pinned), 1);  pin_var = pin_val;
    if reg_col(vi) == 0, return; end
    h = reg_col(vi);  k = find(pinned == h);  cs = cs(:);  s = ts(vi);
    xiB  = XiN(free, vi) * s ./ cs(free);
    varB = XiN_var(free, vi) * s^2 ./ cs(free).^2;
    H = Theta(:, h);
    if strcmp(mode, 'drift')
        D_R = mean(H);
        xi_R = -mean(Theta(:, free) * xiB) / D_R;
        g = mean(Theta(:, free), 1)' / D_R;
    else
        r = dXdt(:, vi) - Theta(:, free) * xiB;
        xi_R = (H' * r) / (H' * H);
        g = (Theta(:, free)' * H) / (H' * H);
    end
    pin_val(k) = max(xi_R, 0) * cs(h) / s;
    pin_var(k) = sum(g.^2 .* varB) * cs(h)^2 / s^2;
end

function xi = stlsq_bounded(Theta, dxdt, lb, ub, lambda, keep)
    % sequentially thresholded least squares with lsqlin bounds; columns in
    % keep are never thresholded
    o = optimoptions('lsqlin', 'Display', 'none');
    active = true(size(Theta, 2), 1);
    for k = 1:15
        xi = zeros(size(Theta, 2), 1);
        xi(active) = lsqlin(Theta(:, active), dxdt, [], [], [], [], lb(active), ub(active), [], o);
        new_active = abs(xi) >= lambda & active;
        new_active(keep) = true;
        if isequal(new_active, active), break; end
        active = new_active;
    end
    xi(~active) = 0;
end

function [score, Xi] = nullcline_score(off, t, X, dXdt, libfun_kn, dvar, tau, k0, n, lbH, ubH, lambda, keepv)
    % bounded SINDy on data shifted by off; fit on the first 70% of the rows,
    % score on the last 30%
    Th = libfun_kn(X(:,1) + off(1), X(:,2) + off(2), get_delayed(t, X(:,dvar), tau) + off(dvar), k0, n);
    N = size(Th, 1);  tr = 1:round(0.7*N);  va = round(0.7*N)+1:N;
    Xi = zeros(size(Th, 2), size(dXdt, 2));
    for v = 1:size(dXdt, 2), Xi(:,v) = stlsq_bounded(Th(tr,:), dXdt(tr,v), lbH{v}, ubH{v}, lambda, nonzeros(keepv(v))); end
    res = dXdt(va,:) - Th(va,:)*Xi;
    R2  = 1 - sum(res.^2, 'all') / max(sum((dXdt(va,:) - mean(dXdt(va,:), 1)).^2, 'all'), 1e-8);
    score = (1 - R2) + 0.01*nnz(Xi);
end

function [W, k0, n] = sr3_fit(t, X, dXdt, libfun_kn, dvar, tau, lbH, ubH, rv, lambda, sr3, fixed, k0_fix, n_fix)
    % relax-and-split: bounded least squares pulled toward a sparse copy W,
    % hard threshold on W (Hill exempt in its equation), and, unless fixed,
    % a nonlinear least-squares update of the Hill parameters (K0, n)
    nu = 0.05;  ts = vecnorm(dXdt, 2, 1);  d = get_delayed(t, X(:,dvar), tau);
    if fixed, k0 = k0_fix;  n = n_fix; else, k0 = sr3.start(1);  n = sr3.start(2); end
    lo = optimoptions('lsqlin', 'Display', 'none');
    no = optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', 60);
    p = numel(lbH{1});  W = zeros(p, numel(lbH));
    for outer = 1:15
        Th  = libfun_kn(X(:,1), X(:,2), d, k0, n);
        csS = vecnorm(Th, 2, 1);  csS = max(csS, 1e-3*median(csS));   % colscale floor
        ThN = Th ./ csS;  Xi = zeros(size(W));
        for v = 1:numel(lbH)
            xiN = lsqlin([ThN; eye(p)/sqrt(nu)], [dXdt(:,v)/ts(v); (W(:,v).*csS'/ts(v))/sqrt(nu)], ...
                         [], [], [], [], lbH{v}.*csS/ts(v), ubH{v}.*csS/ts(v), [], lo);
            Xi(:,v) = xiN * ts(v) ./ csS';
        end
        W = Xi;  W(abs(W) < lambda) = 0;  W(p,rv) = Xi(p,rv);
        if fixed, continue; end
        best = inf;  starts = [k0, n; sr3.p_lb + rand(3,2).*(sr3.p_ub - sr3.p_lb)];
        for s = 1:size(starts, 1)
            [q, cost] = lsqnonlin(@(q) sr3_residual(q, X, d, dXdt, Xi, libfun_kn), starts(s,:), sr3.p_lb, sr3.p_ub, no);
            if cost < best, best = cost;  k0 = q(1);  n = q(2); end
        end
    end
end

function res = sr3_residual(q, X, d, dXdt, Xi, libfun_kn)
    % model residual with Hill parameters q = [K0, n]; the last entry keeps
    % the Hill column from vanishing
    Th  = libfun_kn(X(:,1), X(:,2), d, q(1), q(2));
    res = [reshape(Th*Xi - dXdt, [], 1); 5/max(norm(Th(:,end)), 1e-6)];
end

function sh = hill_share(Theta, xi, h)
    % share of the equation's mean production carried by the Hill term
    contrib = xi' .* mean(Theta, 1);
    pos = contrib(contrib > 0);
    sh = 0;  if ~isempty(pos), sh = max(contrib(h), 0) / sum(pos); end
end

function X = simulate(Xi, libfun, t, Xdata, dvar, tau, dt, T, off)
    % delay-aware Euler step from the first data point; the measured delayed
    % variable is the history for t < tau; off is a state offset of the library
    N = round(T/dt) + 1;  L = round(tau/dt);
    X = nan(N, 2);  X(1,:) = Xdata(1,:);
    for k = 2:N
        if k-1-L >= 1, d = X(k-1-L, dvar); else, d = interp1(t, Xdata(:,dvar), max((k-2)*dt - tau, 0)); end
        X(k,:) = X(k-1,:) + dt * (libfun(X(k-1,1) + off(1), X(k-1,2) + off(2), d + off(dvar)) * Xi);
        if any(~isfinite(X(k,:))) || any(abs(X(k,:)) > 1e3), X(k:end,:) = NaN;  break; end
    end
end

function s = score_model(Xsim, Xd, dt, osc_frac)
    % behaviour of a free-running model against the data window
    W = Xsim(1:size(Xd,1), :);
    s.rmse    = sqrt(mean((W - Xd).^2, 1));
    s.rel     = vecnorm(W - Xd) ./ vecnorm(Xd);
    s.rel_sys = norm(W - Xd, 'fro') / norm(Xd, 'fro');       % system-wide, both variables
    s.level   = mean(W, 1) ./ mean(Xd, 1) - 1;
    s.amp     = std(W, 0, 1) ./ std(Xd, 0, 1);
    late      = Xsim(round(end/2):end, :);
    s.sustained = all(isfinite(late(:))) && all(std(late, 0, 1) > osc_frac * std(Xd, 0, 1));
    s.period = NaN;  s.lag = NaN;
    if isequal(Xsim, Xd), late = Xd; end                    % the data themselves: whole window
    if s.sustained || isequal(Xsim, Xd)
        u = late(:,1) - mean(late(:,1));  w = late(:,2) - mean(late(:,2));
        [ac, lg] = xcorr(u, min(round(20/dt), numel(u) - 1), 'coeff');     % period: first
        ac = ac(lg > 0);  lg = lg(lg > 0);  z = find(ac < 0, 1);           % autocorrelation peak
        if ~isempty(z), [~, i] = max(ac(z:end));  s.period = lg(z + i - 1) * dt; end
        [cc, lags] = xcorr(w, u, round(6/dt), 'coeff');
        cc(lags < 0) = -inf;  [~, ix] = max(cc);  s.lag = lags(ix) * dt;
    end
end

function [period, gain] = linear_mode(Xi, libfun, dvar, tau, scale)
    % Oscillatory mode of the model linearized at its fixed point:
    % det(s I - A - B exp(-s tau)) = 0, A = dF/dz, B = dF/dz(t - tau).
    % Returns the period 2 pi / imag(s) and the amplitude multiplier per
    % period exp(real(s) period) of the complex root with the largest real
    % part (NaN if there is none).
    period = NaN;  gain = NaN;
    rhs = @(z, d) libfun(z(1), z(2), d) * Xi;
    o = optimoptions('lsqnonlin', 'Display', 'off');
    best = inf;  zs = [NaN NaN];
    for a = [0.3 1 2]
        for b = [0.3 1 2]
            [z, r] = lsqnonlin(@(z) rhs(z, z(dvar))', [a b] .* scale, [0 0], 10*scale, o);
            if r < best, best = r;  zs = z; end
        end
    end
    if best > 1e-10 * max(1, max(scale))^2, return; end
    h = 1e-6 * max(1, max(scale));  A = zeros(2);  B = zeros(2);
    for j = 1:2
        e = zeros(1, 2);  e(j) = h;
        A(:,j) = (rhs(zs + e, zs(dvar)) - rhs(zs - e, zs(dvar)))' / (2*h);
    end
    B(:,dvar) = (rhs(zs, zs(dvar) + h) - rhs(zs, zs(dvar) - h))' / (2*h);
    ce = @(s) det((s(1) + 1i*s(2))*eye(2) - A - B*exp(-(s(1) + 1i*s(2))*tau));
    found = [];
    for re0 = [-2 -1 -0.3 0 0.3]
        for im0 = [0.5 1 1.5 2.5 4]
            [s, r] = lsqnonlin(@(s) [real(ce(s)); imag(ce(s))], [re0 im0], [], [], o);
            if r < 1e-12 && abs(s(2)) > 1e-3, found(end+1) = s(1) + 1i*abs(s(2)); end %#ok<AGROW>
        end
    end
    if isempty(found), return; end
    [~, k] = max(real(found));
    period = 2*pi / imag(found(k));
    gain   = exp(real(found(k)) * period);
end
