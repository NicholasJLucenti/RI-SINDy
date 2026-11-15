%% =========================================================================
%  p53-Mdm2 single-cell oscillations -- RI-SINDy and bounded SINDy
%
%  Data: six single-cell p53-CFP / Mdm2-YFP traces digitized from
%  Geva-Zatorsky et al. (2006) Mol Syst Biol 2:2006.0033, Fig 1B
%  ("p53 Data/", min-max normalized per cell, about 31 h).
%
%  Model class, fitted to each cell separately (x = p53, y = Mdm2):
%      dx/dt = terms of [1, x, x^2, y, y^2, x*y]
%      dy/dt = terms of [1, x, x^2, y, y^2, x*y] + xi_R * Hill(x(t - tau))
%  The delayed Hill activation of Mdm2 by p53 is the regulatory term.
%
%  Both methods use the same library and the same lsqlin bounds (BOUNDS
%  block below). They differ only in the Hill coefficient: bounded SINDy
%  regresses it with the polynomial terms, RI-SINDy sets it by the drift
%  balance (xi_R = -D_B/D_R, Eq. 25, with the sign bound xi_R >= 0).
%
%  Output (console): identified equations per cell with 95% credible
%  intervals (RI-SINDy), convergence per cell and the stopping rule, the
%  sign of the unclipped xi_R, library condition numbers, and the
%  behaviour of each identified model run on its own (mean level,
%  amplitude, sustained oscillation, period, Mdm2 lag, damping per cycle).
%  Figures: (1)-(2) identified model vs data, one per method;
%           (3) free-running p53 over 120 h; (4) convergence traces.
%
%  The same algorithm block (local functions at the bottom) is used by
%  every Hes1 and p53 script.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr). No random numbers are used.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));
dataDir = fullfile(thisDir, '..', '..', 'p53 Data');

%% --- SETTINGS -------------------------------------------------------------
n_cells     = 6;
dt          = 0.05;     % h; interpolation grid
sg_window   = 7;        % Savitzky-Golay window on the 0.25 h digitized grid
tau         = 1.0;      % h; delay of p53 -> Mdm2 activation
hill_k0     = 0.4;      % Hill half-saturation, normalized units (data are 0-1)
hill_n      = 4;        % Hill exponent (p53 binds DNA as a tetramer)
t_half_mdm2 = 0.5;      % h; Mdm2 half-life prior
t_half_p53  = 0.5;      % h; p53 half-life prior at the mean Mdm2 level
lambda      = 0.05;     % |xi| below this is treated as zero (every method)
T_sim       = 120;      % h; free run of each identified model
osc_frac    = 0.2;      % sustained = SD over the last half > osc_frac x data SD
% RI-SINDy iteration settings (identical in every Hes1 and p53 script)
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;          % stopping rule: converged if xi_R moves < 1% from K = 60 to 100

col_names = {'1','x','x^2','y','y^2','x*y','Hill'};
reg_col   = [0, 7];     % regulatory column per equation (0 = none)
hill      = @(d) max(d, 0).^hill_n ./ (hill_k0^hill_n + max(d, 0).^hill_n);
libfun    = @(x, y, d) [ones(size(x)), x, x.^2, y, y.^2, x.*y, hill(d)];
dvar      = 1;          % the delayed variable is p53

labels = {'RI-SINDy', 'Bounded SINDy'};
Xi_all = zeros(7, 2, n_cells, 2);  sd_ri = zeros(7, 2, n_cells);
ok = true(n_cells, 2);  xiR_raw = zeros(n_cells, 1);  kappa = zeros(n_cells, 2);
S = cell(n_cells, 2);  Sd = cell(n_cells, 1);  sim = cell(n_cells, 2);  data = cell(n_cells, 1);
mode_T = nan(n_cells, 2);  mode_g = nan(n_cells, 2);  traces = cell(n_cells, 1);
iters = [1:10, 12:3:30, 35:5:K_check];

%% --- FIT EVERY CELL -------------------------------------------------------
for c = 1:n_cells
    A = readmatrix(fullfile(dataDir, sprintf('p53_cell%d.csv', c)));   % time_h, p53, Mdm2
    t = (0:dt:A(end,1))';
    x = max(interp1(A(:,1), sgolayfilt(A(:,2), 3, sg_window), t, 'spline'), 0);
    y = max(interp1(A(:,1), sgolayfilt(A(:,3), 3, sg_window), t, 'spline'), 0);
    dXdt  = [smooth_derivative(x, dt, 3, 11), smooth_derivative(y, dt, 3, 11)];
    Theta = libfun(x, y, get_delayed(t, x, tau));
    data{c} = struct('t', t, 'X', [x, y]);

    % ---- BOUNDS on physical coefficients, columns [1, x, x^2, y, y^2, x*y] ----
    % p53:  synthesis >= 0; every other term <= 0 (degradation); the
    %       Mdm2-dependent degradation x*y is at least as fast as the p53
    %       half-life at the cell's mean Mdm2 level.
    % Mdm2: no constant (a free constant makes the drift balance
    %       non-identifying); x, x^2 >= 0 (p53-driven production);
    %       y decays at least as fast as the Mdm2 half-life; y^2, x*y <= 0.
    lb = {[0, -inf, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf, -inf]};
    ub = {[inf, 0, 0, 0, 0, -log(2)/(t_half_p53*mean(y))], [0, inf, inf, -log(2)/t_half_mdm2, 0, 0]};
    % the same bounds with the Hill column appended, for the regression baselines
    lbH = {[lb{1}, 0], [lb{2}, 0]};   ubH = {[ub{1}, 0], [ub{2}, inf]};

    % ---- RI-SINDy ----
    [Xi, Xi_var, ok(c,1)] = fit_risindy(Theta, dXdt, lb, ub, reg_col, 'drift', ri_opts, K_check);
    xiR_raw(c) = -mean(Theta(:,1:6) * Xi(1:6,2)) / mean(Theta(:,7));   % before the sign bound
    Xi(abs(Xi) < lambda) = 0;
    Xi_all(:,:,c,1) = Xi;  sd_ri(:,:,c) = sqrt(Xi_var) .* (Xi ~= 0);

    % ---- Bounded SINDy: sequentially thresholded lsqlin, Hill regressed ----
    Xi_all(:,1,c,2) = stlsq_bounded(Theta, dXdt(:,1), lbH{1}, ubH{1}, lambda, []);
    Xi_all(:,2,c,2) = stlsq_bounded(Theta, dXdt(:,2), lbH{2}, ubH{2}, lambda, 7);

    % ---- conditioning of the unit-norm library, with and without the Hill column ----
    ThN = Theta ./ vecnorm(Theta);
    kappa(c,:) = [cond(ThN(:,1:6)), cond(ThN)];

    % ---- behaviour of each identified model run on its own ----
    Sd{c} = score_model([x, y], [x, y], dt, osc_frac);
    for m = 1:2
        if ~ok(c,m), continue; end
        sim{c,m} = simulate(Xi_all(:,:,c,m), libfun, t, [x, y], dvar, tau, dt, T_sim, [0 0]);
        S{c,m}   = score_model(sim{c,m}, [x, y], dt, osc_frac);
        [mode_T(c,m), mode_g(c,m)] = linear_mode(Xi_all(:,:,c,m), libfun, dvar, tau, mean([x, y]));
    end

    % ---- convergence trace of xi_R (RI-SINDy) ----
    tr = zeros(size(iters));
    for k = 1:numel(iters)
        o = ri_opts;  o.n_iter = iters(k);
        Xk = fit_risindy(Theta, dXdt, lb, ub, reg_col, 'drift', o, 0);
        tr(k) = Xk(7,2);
    end
    traces{c} = tr;
    fprintf('cell %d fitted\n', c);
end

%% --- RESULTS: identified equations ----------------------------------------
fprintf('\nRI-SINDy stopping rule: K_max = %d iterations; converged if xi_R changes by < 1%% between K = %d and K = %d.\n', ...
        ri_opts.n_iter, ri_opts.n_iter, K_check);
fprintf('Settings: tau = %.2f h, K0 = %.2f, n = %d, half-lives %.0f / %.0f min, lambda = %.2f, eta = %.1f, prior %s\n', ...
        tau, hill_k0, hill_n, 60*t_half_p53, 60*t_half_mdm2, lambda, ri_opts.eta_pin, ri_opts.prior);
for m = 1:2
    fprintf('\n=== %s: identified model per cell ===\n', labels{m});
    for c = 1:n_cells
        eq = {'', ''};
        for v = 1:2
            for k = 1:7
                if Xi_all(k,v,c,m) == 0, continue; end
                if m == 1
                    eq{v} = sprintf('%s %+.3f [%+.3f, %+.3f] %s', eq{v}, Xi_all(k,v,c,m), ...
                        Xi_all(k,v,c,m) - 1.96*sd_ri(k,v,c), Xi_all(k,v,c,m) + 1.96*sd_ri(k,v,c), col_names{k});
                else
                    eq{v} = sprintf('%s %+.3f %s', eq{v}, Xi_all(k,v,c,m), col_names{k});
                end
            end
        end
        note = '';
        if m == 1 && ~ok(c,1), note = '   [did not converge]'; end
        fprintf('cell %d  d(p53)/dt  =%s\n        d(Mdm2)/dt =%s%s\n', c, eq{1}, eq{2}, note);
    end
end
fprintf('\n(RI-SINDy brackets: 95%% credible interval, coefficient +/- 1.96 posterior SD)\n');

fprintf('\n=== Mdm2 equation across cells: mean +/- SD (cells where the term is nonzero) ===\n');
fprintf('%-6s %26s %26s\n', 'term', labels{:});
for k = 2:7
    s = cell(1, 2);
    for m = 1:2
        v = squeeze(Xi_all(k, 2, ok(:,m), m));
        s{m} = sprintf('%+.2f +/- %.2f  (%d/%d)', mean(v), std(v), nnz(v), numel(v));
    end
    fprintf('%-6s %26s %26s\n', col_names{k}, s{:});
end

fprintf('\nSign of xi_R before the bound xi_R >= 0: min %.3f, max %.3f over cells (bound active in %d cells)\n', ...
        min(xiR_raw), max(xiR_raw), sum(xiR_raw < 0));
fprintf('Library condition number (unit-norm columns), mean over cells: without Hill %.1f, with Hill %.1f (ratio %.1f)\n', ...
        mean(kappa(:,1)), mean(kappa(:,2)), mean(kappa(:,2)./kappa(:,1)));

%% --- RESULTS: behaviour of the identified models --------------------------
fprintf('\n=== Identified model run on its own from the first data point ===\n');
fprintf('%-6s %-14s %8s %8s %8s %8s %9s %8s %8s %8s %9s\n', 'cell', 'method', 'lvl p53', 'lvl Mdm2', ...
        'amp p53', 'amp Mdm2', 'sustained', 'period', 'lag', 'RMSE p53', 'x/cycle');
for c = 1:n_cells
    fprintf('%-6d %-14s %8s %8s %8s %8s %9s %8.1f %8.1f %8s %9s\n', c, 'data', '', '', '', '', '', Sd{c}.period, Sd{c}.lag, '', '');
    for m = 1:2
        if ~ok(c,m), fprintf('%-6s %-14s did not converge\n', '', labels{m}); continue; end
        s = S{c,m};
        fprintf('%-6s %-14s %+7.0f%% %+7.0f%% %8.2f %8.2f %9d %8.1f %8.1f %8.3f %9.2f\n', '', labels{m}, ...
                100*s.level(1), 100*s.level(2), s.amp(1), s.amp(2), s.sustained, s.period, s.lag, s.rmse(1), mode_g(c,m));
    end
end
fprintf('\n(lvl: mean of the model over the data window relative to the data mean; amp: model SD / data SD;\n');
fprintf(' sustained: SD over %d-%d h above %.1f x data SD; x/cycle: amplitude multiplier per cycle of the\n', T_sim/2, T_sim, osc_frac);
fprintf(' linearized model at its fixed point, < 1 damped, > 1 self-sustained)\n');

fprintf('\n=== Summary over cells (RI-SINDy: converged cells only) ===\n');
fprintf('%-38s %16s %16s\n', '', labels{:});
pick = @(m, f) cellfun(@(s) s.(f), S(ok(:,m), m), 'UniformOutput', false);
summ = @(m, f, j) cellfun(@(v) v(j), pick(m, f));
ms   = @(v) sprintf('%.2f +/- %.2f', mean(v, 'omitnan'), std(v, 'omitnan'));
fprintf('%-38s %16s %16s\n', 'Fits converged', sprintf('%d / %d', sum(ok(:,1)), n_cells), sprintf('%d / %d', sum(ok(:,2)), n_cells));
fprintf('%-38s %16s %16s\n', 'Cells with sustained oscillation', sprintf('%d / %d', sum(summ(1,'sustained',1)), n_cells), ...
        sprintf('%d / %d', sum(summ(2,'sustained',1)), n_cells));
fprintf('%-38s %16s %16s\n', '|mean level error| p53 (%)',  ms(100*abs(summ(1,'level',1))), ms(100*abs(summ(2,'level',1))));
fprintf('%-38s %16s %16s\n', '|mean level error| Mdm2 (%)', ms(100*abs(summ(1,'level',2))), ms(100*abs(summ(2,'level',2))));
fprintf('%-38s %16s %16s\n', 'Amplitude ratio p53',  ms(summ(1,'amp',1)), ms(summ(2,'amp',1)));
fprintf('%-38s %16s %16s\n', 'Amplitude ratio Mdm2', ms(summ(1,'amp',2)), ms(summ(2,'amp',2)));
fprintf('%-38s %16s %16s\n', 'Amplitude multiplier per cycle', ms(mode_g(ok(:,1),1)), ms(mode_g(ok(:,2),2)));
fprintf('%-38s %16s %16s\n', 'Period, sustained cells (h)', ms(summ(1,'period',1)), ms(summ(2,'period',1)));
fprintf('%-38s %16s %16s\n', 'Pointwise RMSE p53',  ms(summ(1,'rmse',1)), ms(summ(2,'rmse',1)));
fprintf('%-38s %16s %16s\n', 'Pointwise RMSE Mdm2', ms(summ(1,'rmse',2)), ms(summ(2,'rmse',2)));
fprintf('Data: period %s h, Mdm2 lag %s h\n', ms(cellfun(@(s) s.period, Sd)), ms(cellfun(@(s) s.lag, Sd)));

%% --- FIGURES (1)-(2): data and the identified model, one figure per method -
for m = 1:2
    figure('Color', 'w', 'Position', [60 + 40*m, 60 + 40*m, 1200, 560]);
    for c = 1:n_cells
        subplot(2, 3, c);  hold on;
        plot(data{c}.t, data{c}.X(:,1), '-', 'Color', [0.1 0.6 0.2], 'LineWidth', 1.2);
        plot(data{c}.t, data{c}.X(:,2), '-', 'Color', [0.8 0.2 0.6], 'LineWidth', 1.2);
        if ok(c,m)
            ts = (0:size(sim{c,m},1)-1)'*dt;
            plot(ts, sim{c,m}(:,1), '--', 'Color', [0.1 0.6 0.2], 'LineWidth', 1.2);
            plot(ts, sim{c,m}(:,2), '--', 'Color', [0.8 0.2 0.6], 'LineWidth', 1.2);
        end
        xlim([0 48]);  ylim([0 1.2]);  box on;
        xlabel('Time (h)');  ylabel('normalized level');  title(sprintf('cell %d', c));
        if c == 1, legend({'p53 data', 'Mdm2 data', 'p53 model', 'Mdm2 model'}, 'FontSize', 7); end
    end
    sgtitle(sprintf('%s: identified model vs single-cell data', labels{m}));
end

%% --- FIGURE (3): free-running p53, both methods ---------------------------
figure('Color', 'w', 'Position', [80 80 1200 560]);
colors = {[0 0.2 0.8], [0.85 0.25 0]};
for c = 1:n_cells
    subplot(2, 3, c);  hold on;
    for m = 1:2
        if ok(c,m), plot((0:size(sim{c,m},1)-1)'*dt, sim{c,m}(:,1), '-', 'Color', colors{m}, 'LineWidth', 1.1); end
    end
    xlim([0 T_sim]);  ylim([0 1]);  box on;
    xlabel('Time (h)');  ylabel('p53 (model)');  title(sprintf('cell %d', c));
    if c == 1, legend(labels, 'FontSize', 7); end
end
sgtitle('Free-running identified models: p53');

%% --- FIGURE (4): convergence of xi_R --------------------------------------
figure('Color', 'w', 'Position', [100 100 560 380]);
for c = 1:n_cells
    semilogy(iters, abs(traces{c} - traces{c}(end)) / max(abs(traces{c}(end)), 1e-12) + 1e-8, '-', 'LineWidth', 1.1);  hold on;
end
xline(ri_opts.n_iter, 'k--');  yline(0.01, 'k:');
xlabel('iteration K');  ylabel('|\xi_R(K) - \xi_R(100)| / |\xi_R(100)|');
legend([arrayfun(@(c) sprintf('cell %d', c), 1:n_cells, 'UniformOutput', false), {'K_{max}', '1%'}], 'FontSize', 7);
title('p53-Mdm2: convergence of the regulatory coefficient');  box on;


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
