%% =========================================================================
%  p53-Mdm2 supplementary test -- RI-SINDy against SINDy variants and ablations
%
%  Methods, fitted to each of the six cells separately (fit_all_methods):
%    1 RI-SINDy          Hill coefficient from the drift balance
%    2 RI loop, LS Hill  ablation: same loop, Hill kept out of the sparse
%                        regression, but its coefficient set by least
%                        squares on the residual instead of the drift balance
%    3 Bounded SINDy     sequentially thresholded lsqlin, Hill regressed
%                        with the polynomial terms (never thresholded)
%    4 Hill removed      ablation: bounded SINDy with no Hill column
%    5 Ensemble-SINDy    bounded SINDy on 100 bootstrap resamples; a term is
%                        kept if it survives in at least half (Fasel 2022)
%    6 Nullcline-SINDy   bounded SINDy after a phase-space offset chosen on
%                        held-out rows (Prokop et al. 2024, adapted)
%    7 SR3, K0 n search  relax-and-split with a parametric Hill column that
%                        searches (K0, n) from an uninformed start (Champion 2020)
%    8 SR3, K0 n fixed   the same, with (K0, n) fixed to RI-SINDy's values
%    9 Traditional SINDy plain STLSQ, no bounds, Hill an ordinary candidate
%
%  Fairness: methods 1-8 use the same library and the same lsqlin bounds
%  (BOUNDS block, identical to p53_risindy.m); the Hill term is exempt from
%  thresholding in the Mdm2 equation for 1-8. Every method uses lambda.
%
%  Scored per method: converged fits; cells whose Mdm2 equation is exactly
%  {y, Hill}; share of Mdm2 production carried by the Hill term; cells
%  whose free-running model sustains an oscillation; amplitude multiplier
%  per cycle; amplitude ratio and mean-level error; RMSE; seconds per cell.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr). Runtime: about 2 minutes.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', '..', 'src')));
dataDir = fullfile(thisDir, '..', '..', '..', 'p53 Data');

%% --- SETTINGS (same as p53_risindy.m) -------------------------------------
rng(42);                % Ensemble-SINDy bootstraps and SR3 restarts
n_cells = 6;  dt = 0.05;  sg_window = 7;
tau = 1.0;  hill_k0 = 0.4;  hill_n = 4;
t_half_mdm2 = 0.5;  t_half_p53 = 0.5;  lambda = 0.05;
T_sim = 120;  osc_frac = 0.2;
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;
sr3 = struct('start', [0.5 2], 'p_lb', [0.05 1], 'p_ub', [1 10]);   % SR3 search for (K0, n)

col_names = {'1','x','x^2','y','y^2','x*y','Hill'};
reg_col   = [0, 7];
libfun_kn = @(x, y, d, k0, n) [ones(size(x)), x, x.^2, y, y.^2, x.*y, max(d,0).^n ./ (k0^n + max(d,0).^n)];
dvar      = 1;
target    = logical([0 0 0 1 0 0 1]);    % Mdm2 equation {y, Hill}

labels = {'RI-SINDy', 'RI loop, LS Hill', 'Bounded SINDy', 'Hill removed', 'Ensemble-SINDy', ...
          'Nullcline-SINDy', 'SR3, K0 n search', 'SR3, K0 n fixed', 'Traditional SINDy'};
n_m = numel(labels);
ok = false(n_cells, n_m);  exact = false(n_cells, n_m);  share = nan(n_cells, n_m);
sust = false(n_cells, n_m);  gain = nan(n_cells, n_m);  secs = zeros(n_cells, n_m);
ampx = nan(n_cells, n_m);  ampy = ampx;  lvlx = ampx;  lvly = ampx;  rmx = ampx;  rmy = ampx;
Xi_store = zeros(7, 2, n_cells, n_m);  k0_store = nan(n_cells, n_m);  n_store = k0_store;

for c = 1:n_cells
    A = readmatrix(fullfile(dataDir, sprintf('p53_cell%d.csv', c)));
    t = (0:dt:A(end,1))';
    x = max(interp1(A(:,1), sgolayfilt(A(:,2), 3, sg_window), t, 'spline'), 0);
    y = max(interp1(A(:,1), sgolayfilt(A(:,3), 3, sg_window), t, 'spline'), 0);
    X = [x, y];
    dXdt = [smooth_derivative(x, dt, 3, 11), smooth_derivative(y, dt, 3, 11)];

    % BOUNDS on physical coefficients, columns [1, x, x^2, y, y^2, x*y] (see p53_risindy.m)
    lb = {[0, -inf, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf, -inf]};
    ub = {[inf, 0, 0, 0, 0, -log(2)/(t_half_p53*mean(y))], [0, inf, inf, -log(2)/t_half_mdm2, 0, 0]};

    R = fit_all_methods(t, X, dXdt, libfun_kn, dvar, tau, hill_k0, hill_n, lb, ub, reg_col, ...
                        lambda, ri_opts, K_check, sr3, true(1, n_m));
    for m = 1:n_m
        ok(c,m) = R.ok(m);  secs(c,m) = R.secs(m);
        Xi = R.Xi{m};  Xi_store(:,:,c,m) = Xi;  k0_store(c,m) = R.k0(m);  n_store(c,m) = R.n(m);
        if ~ok(c,m), continue; end
        lib = @(a, b, d) libfun_kn(a, b, d, R.k0(m), R.n(m));
        Th  = lib(x + R.off(m,1), y + R.off(m,2), get_delayed(t, x, tau) + R.off(m,dvar));
        exact(c,m) = isequal(Xi(:,2)' ~= 0, target);
        share(c,m) = hill_share(Th, Xi(:,2), 7);
        Xs = simulate(Xi, lib, t, X, dvar, tau, dt, T_sim, R.off(m,:));
        s  = score_model(Xs, X, dt, osc_frac);
        sust(c,m) = s.sustained;  ampx(c,m) = s.amp(1);  ampy(c,m) = s.amp(2);
        lvlx(c,m) = s.level(1);  lvly(c,m) = s.level(2);  rmx(c,m) = s.rmse(1);  rmy(c,m) = s.rmse(2);
        if all(R.off(m,:) == 0), [~, gain(c,m)] = linear_mode(Xi, lib, dvar, tau, mean(X)); end
    end
    fprintf('cell %d done\n', c);
end

%% --- RESULTS --------------------------------------------------------------
fprintf('\nIdentified Mdm2 equation, d(Mdm2)/dt, per cell\n');
for m = 1:n_m
    fprintf('\n%s\n', labels{m});
    for c = 1:n_cells
        if ~ok(c,m), fprintf('  cell %d: did not converge\n', c); continue; end
        terms = '';
        for k = 1:7
            if Xi_store(k,2,c,m) ~= 0, terms = sprintf('%s %+.2f*%s', terms, Xi_store(k,2,c,m), col_names{k}); end
        end
        extra = '';
        if m == 7, extra = sprintf('   [found K0 = %.2f, n = %.1f]', k0_store(c,m), n_store(c,m)); end
        fprintf('  cell %d:%s%s\n', c, terms, extra);
    end
end

ms  = @(v) sprintf('%.2f+/-%.2f', mean(v, 'omitnan'), std(v, 'omitnan'));
cnt = @(M) arrayfun(@(m) sprintf('%d/%d', sum(M(:,m)), n_cells), 1:n_m, 'UniformOutput', false);
avg = @(M) arrayfun(@(m) ms(M(ok(:,m),m)), 1:n_m, 'UniformOutput', false);
fprintf('\n%-32s', '');  fprintf('%-18s', labels{:});  fprintf('\n');
row = @(name, v) fprintf(['%-32s' repmat('%-18s', 1, n_m) '\n'], name, v{:});
row('Fits converged',                    cnt(ok));
row('Mdm2 eq is exactly {y, Hill}',      cnt(exact));
row('Hill share of Mdm2 production',     avg(share));
row('Cells with sustained oscillation',  cnt(sust));
row('Amplitude multiplier per cycle',    avg(gain));
row('Amplitude ratio p53',               avg(ampx));
row('Amplitude ratio Mdm2',              avg(ampy));
row('|Mean level error| p53',            avg(abs(lvlx)));
row('|Mean level error| Mdm2',           avg(abs(lvly)));
row('Pointwise RMSE p53',                avg(rmx));
row('Pointwise RMSE Mdm2',               avg(rmy));
row('Seconds per cell',                  arrayfun(@(m) sprintf('%.2f', mean(secs(:,m))), 1:n_m, 'UniformOutput', false));
fprintf('(amplitude multiplier: linearized model at its fixed point; not computed for Nullcline-SINDy, whose model lives on offset coordinates)\n');

%% --- FIGURE ---------------------------------------------------------------
figure('Color', 'w', 'Position', [60 60 1300 420]);
panels = {sum(exact, 1),                              'Cells with Mdm2 eq = \{y, Hill\}', [0 n_cells];
          sum(sust, 1),                               'Cells with sustained oscillation', [0 n_cells];
          arrayfun(@(m) mean(ampx(ok(:,m),m), 'omitnan'), 1:n_m), 'Amplitude ratio p53 (model / data)', [0 1];
          arrayfun(@(m) mean(share(ok(:,m),m), 'omitnan'), 1:n_m), 'Hill share of Mdm2 production', [0 1]};
for q = 1:4
    subplot(1, 4, q);
    bar(panels{q,1}, 'FaceColor', [0.3 0.45 0.7]);
    set(gca, 'XTick', 1:n_m, 'XTickLabel', labels, 'XTickLabelRotation', 45, 'FontSize', 7);
    ylim(panels{q,3});  title(panels{q,2}, 'FontWeight', 'normal', 'FontSize', 8);  box on;
end
sgtitle('p53-Mdm2: methods and ablations across six cells');


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
