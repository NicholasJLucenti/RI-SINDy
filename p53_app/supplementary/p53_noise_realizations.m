%% =========================================================================
%  p53-Mdm2 supplementary test -- repeated noise realizations, every method
%
%  Adds measurement noise to each digitized trace (noise_level x SD of the
%  trace, per variable, at the 0.25 h sample times), repeats the fits of
%  p53_method_comparison.m (same nine methods, library and lsqlin bounds)
%  n_rep times per cell and noise level, and reports mean +/- SD over all
%  cells and realizations of
%    - % of fits whose Mdm2 equation is exactly {y, Hill}
%    - % of fits whose free-running model sustains an oscillation
%    - share of Mdm2 production carried by the Hill term
%    - p53 amplitude ratio (model SD / data SD)
%    - RMSE per variable and system-wide relative L2 error (both variables
%      stacked), against the noise-free digitized trace
%  A RI-SINDy fit that does not converge counts as a failure in the rates
%  and is left out of the averages.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr). Runtime: about 20 minutes.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', '..', 'src')));
dataDir = fullfile(thisDir, '..', '..', '..', 'p53 Data');

%% --- SETTINGS (same as p53_risindy.m) -------------------------------------
rng(42);                              % noise, bootstraps and SR3 restarts
n_cells = 6;  dt = 0.05;  sg_window = 7;
tau = 1.0;  hill_k0 = 0.4;  hill_n = 4;
t_half_mdm2 = 0.5;  t_half_p53 = 0.5;  lambda = 0.05;
T_sim = 120;  osc_frac = 0.2;
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;
sr3 = struct('start', [0.5 2], 'p_lb', [0.05 1], 'p_ub', [1 10]);
noise_list = [0 0.02 0.05 0.10];      % x SD of each trace
n_rep      = 20;

reg_col   = [0, 7];  dvar = 1;  target = logical([0 0 0 1 0 0 1]);
libfun_kn = @(x, y, d, k0, n) [ones(size(x)), x, x.^2, y, y.^2, x.*y, max(d,0).^n ./ (k0^n + max(d,0).^n)];
labels = {'RI-SINDy', 'RI loop, LS Hill', 'Bounded SINDy', 'Hill removed', 'Ensemble-SINDy', ...
          'Nullcline-SINDy', 'SR3, K0 n search', 'SR3, K0 n fixed', 'Traditional SINDy'};
n_m = numel(labels);  nL = numel(noise_list);
F = struct('n', zeros(nL, n_m), 'exact', zeros(nL, n_m), 'sust', zeros(nL, n_m));
V = struct('share', {cell(nL, n_m)}, 'amp', {cell(nL, n_m)}, 'rx', {cell(nL, n_m)}, ...
           'ry', {cell(nL, n_m)}, 'rel', {cell(nL, n_m)}, 'hill', {cell(nL, n_m)});

%% --- RUN ------------------------------------------------------------------
for l = 1:nL
  for c = 1:n_cells
    A  = readmatrix(fullfile(dataDir, sprintf('p53_cell%d.csv', c)));
    t  = (0:dt:A(end,1))';
    X0 = max([interp1(A(:,1), sgolayfilt(A(:,2), 3, sg_window), t, 'spline'), ...
              interp1(A(:,1), sgolayfilt(A(:,3), 3, sg_window), t, 'spline')], 0);   % reference
    for r = 1:n_rep
        if noise_list(l) == 0 && r > 1, break; end      % no noise: one fit is enough
        xa = A(:,2) + noise_list(l)*std(A(:,2))*randn(size(A,1), 1);
        ya = A(:,3) + noise_list(l)*std(A(:,3))*randn(size(A,1), 1);
        X  = max([interp1(A(:,1), sgolayfilt(xa, 3, sg_window), t, 'spline'), ...
                  interp1(A(:,1), sgolayfilt(ya, 3, sg_window), t, 'spline')], 0);
        dXdt = [smooth_derivative(X(:,1), dt, 3, 11), smooth_derivative(X(:,2), dt, 3, 11)];

        % BOUNDS on physical coefficients, columns [1, x, x^2, y, y^2, x*y] (see p53_risindy.m)
        lb = {[0, -inf, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf, -inf]};
        ub = {[inf, 0, 0, 0, 0, -log(2)/(t_half_p53*mean(X(:,2)))], [0, inf, inf, -log(2)/t_half_mdm2, 0, 0]};

        R = fit_all_methods(t, X, dXdt, libfun_kn, dvar, tau, hill_k0, hill_n, lb, ub, reg_col, ...
                            lambda, ri_opts, K_check, sr3, true(1, n_m));
        for m = 1:n_m
            F.n(l,m) = F.n(l,m) + 1;
            if ~R.ok(m), continue; end
            Xi  = R.Xi{m};  lib = @(a, b, d) libfun_kn(a, b, d, R.k0(m), R.n(m));
            Th  = lib(X(:,1) + R.off(m,1), X(:,2) + R.off(m,2), get_delayed(t, X(:,1), tau) + R.off(m,1));
            s   = score_model(simulate(Xi, lib, t, X, dvar, tau, dt, T_sim, R.off(m,:)), X0, dt, osc_frac);
            F.exact(l,m) = F.exact(l,m) + isequal(Xi(:,2)' ~= 0, target);
            F.sust(l,m)  = F.sust(l,m) + s.sustained;
            V.share{l,m}(end+1) = hill_share(Th, Xi(:,2), 7);
            V.amp{l,m}(end+1) = s.amp(1);   V.rx{l,m}(end+1) = s.rmse(1);
            V.ry{l,m}(end+1)  = s.rmse(2);  V.rel{l,m}(end+1) = s.rel_sys;  V.hill{l,m}(end+1) = Xi(7,2);
        end
    end
  end
  fprintf('noise %.0f%% done\n', 100*noise_list(l));
end

%% --- RESULTS --------------------------------------------------------------
ms = @(v) sprintf('%.2f+/-%.2f', mean(v, 'omitnan'), std(v, 'omitnan'));
fprintf('\n%-18s %6s %8s %8s %13s %13s %13s %13s %13s %13s\n', 'method', 'noise', '{y,Hill}', 'oscil.', ...
        'Hill share', 'amp p53', 'RMSE p53', 'RMSE Mdm2', 'rel L2 (sys)', 'Hill coef');
for m = 1:n_m
    for l = 1:nL
        fprintf('%-18s %5.0f%% %7.0f%% %7.0f%% %13s %13s %13s %13s %13s %13s\n', labels{m}, 100*noise_list(l), ...
                100*F.exact(l,m)/F.n(l,m), 100*F.sust(l,m)/F.n(l,m), ms(V.share{l,m}), ms(V.amp{l,m}), ...
                ms(V.rx{l,m}), ms(V.ry{l,m}), ms(V.rel{l,m}), ms(V.hill{l,m}));
    end
end
fprintf('(rates over %d fits per noise level, %d at 0%%; rel L2 is system-wide: both variables stacked)\n', ...
        n_cells*n_rep, n_cells);

%% --- FIGURE ---------------------------------------------------------------
figure('Color', 'w', 'Position', [60 60 1300 380]);
mk = {'-o', '--s', '-^', ':x', '-.d', '--v', ':>', '-<', '--p'};
pn = {100*F.exact ./ F.n, 'Mdm2 eq = \{y, Hill\} (% of fits)', [0 100];
      100*F.sust  ./ F.n, 'sustained oscillation (% of fits)', [0 100];
      cellfun(@(v) mean(v, 'omitnan'), V.amp), 'amplitude ratio p53 (model / data)', [0 1.2];
      cellfun(@(v) mean(v, 'omitnan'), V.rx),  'RMSE p53', [0 0.6]};
for q = 1:4
    subplot(1, 4, q);  hold on;
    for m = 1:n_m, plot(100*noise_list, pn{q,1}(:,m), mk{m}, 'LineWidth', 1.1); end
    xlabel('added noise (% of trace SD)');  ylabel(pn{q,2});  ylim(pn{q,3});  box on;
    if q == 4, legend(labels, 'FontSize', 6, 'Location', 'best'); end
end
sgtitle('p53-Mdm2: repeated noise realizations');


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
