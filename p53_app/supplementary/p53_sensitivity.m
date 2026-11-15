%% =========================================================================
%  p53-Mdm2 supplementary test -- sensitivity to every assumed setting
%
%  Repeats the fit of paper/p53_app/p53_risindy.m (same library and lsqlin
%  bounds) on all six cells while one pair of settings is swept:
%    (A) Hill half-saturation K0 x Hill exponent n
%    (B) half-life prior (both proteins) x delay tau; the half-life prior
%        sets two lsqlin bounds, so this is the sensitivity to the bounds
%    (C) threshold lambda x relaxation factor eta (eta_pin, Eq. 28)
%    (D) sparsity prior: Spike-and-Slab, Laplace, Horseshoe
%  for RI-SINDy and bounded SINDy (which has no eta and no prior).
%  Mapped: share of Mdm2 production carried by the delayed Hill term; cells
%  whose Mdm2 equation is exactly {y, Hill}; cells whose free-running model
%  sustains an oscillation. A RI-SINDy fit that does not converge is left
%  out of the share and counts as a failure in the counts.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr). Runtime: about 5 minutes. No random numbers.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', '..', 'src')));
dataDir = fullfile(thisDir, '..', '..', '..', 'p53 Data');

%% --- SETTINGS (central values = p53_risindy.m) ----------------------------
n_cells = 6;  dt = 0.05;  sg_window = 7;
tau = 1.0;  hill_k0 = 0.4;  hill_n = 4;  t_half = 0.5;  lambda = 0.05;
T_sim = 120;  osc_frac = 0.2;
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;

k0_list    = 0.2:0.1:0.7;                  % (A) Hill half-saturation (normalized units)
n_list     = 1:6;                          % (A) Hill exponent
thalf_list = [15 20 30 45 60 90];          % (B) half-life prior, both proteins (min)
tau_list   = [0.5 0.75 1.0 1.25 1.5 2.0];  % (B) delay (h)
lam_list   = [0.01 0.02 0.05 0.1 0.2 0.5]; % (C) threshold, log-spaced
eta_list   = [0.1 0.2 0.4 0.6 0.8 1.0];    % (C) relaxation factor
prior_list = {'SpikeSlab', 'Laplace', 'Horseshoe'};   % (D)

reg_col   = [0, 7];  dvar = 1;  target = logical([0 0 0 1 0 0 1]);
libfun_kn = @(x, y, d, k0, n) [ones(size(x)), x, x.^2, y, y.^2, x.*y, max(d,0).^n ./ (k0^n + max(d,0).^n)];

%% --- DATA -----------------------------------------------------------------
D = cell(n_cells, 1);
for c = 1:n_cells
    A = readmatrix(fullfile(dataDir, sprintf('p53_cell%d.csv', c)));
    t = (0:dt:A(end,1))';
    x = max(interp1(A(:,1), sgolayfilt(A(:,2), 3, sg_window), t, 'spline'), 0);
    y = max(interp1(A(:,1), sgolayfilt(A(:,3), 3, sg_window), t, 'spline'), 0);
    D{c} = struct('t', t, 'X', [x, y], 'dXdt', [smooth_derivative(x, dt, 3, 11), smooth_derivative(y, dt, 3, 11)]);
end
% BOUNDS on physical coefficients, columns [1, x, x^2, y, y^2, x*y] (see p53_risindy.m)
bounds = @(y, th) deal({[0, -inf, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf, -inf]}, ...
                       {[inf, 0, 0, 0, 0, -log(2)/(th*mean(y))], [0, inf, inf, -log(2)/th, 0, 0]});
run_cell = @(c, k0, n, tu, th, lam, opts) score_cell(D{c}, k0, n, tu, th, lam, opts, K_check, ...
               libfun_kn, bounds, reg_col, dvar, target, dt, T_sim, osc_frac);

%% --- (A) K0 x n and (B) half-life x tau -----------------------------------
grids = {n_list, k0_list, 'A';  tau_list, thalf_list, 'B'};
Z = struct();
for g = 1:2
    rows = grids{g,1};  cols = grids{g,2};
    sh = nan(numel(rows), numel(cols), 2);  ex = zeros(size(sh));  os = zeros(size(sh));
    for i = 1:numel(rows)
        for j = 1:numel(cols)
            if g == 1, k0 = cols(j);  n = rows(i);  tu = tau;     th = t_half;
            else,      k0 = hill_k0;  n = hill_n; tu = rows(i);  th = cols(j)/60; end
            shc = nan(n_cells, 2);
            for c = 1:n_cells
                r = run_cell(c, k0, n, tu, th, lambda, ri_opts);
                shc(c,:) = r.share;  ex(i,j,:) = ex(i,j,:) + reshape(r.exact, 1, 1, 2);
                os(i,j,:) = os(i,j,:) + reshape(r.sust, 1, 1, 2);
            end
            sh(i,j,:) = mean(shc, 1, 'omitnan');
        end
    end
    Z.(grids{g,3}) = struct('share', sh, 'exact', ex, 'sust', os);
    fprintf('sweep %s done\n', grids{g,3});
end

%% --- (C) lambda x eta -----------------------------------------------------
exC = zeros(numel(eta_list), numel(lam_list));  osC = exC;
exS = zeros(1, numel(lam_list));  osS = exS;
for i = 1:numel(eta_list)
    o = ri_opts;  o.eta_pin = eta_list(i);
    for j = 1:numel(lam_list)
        for c = 1:n_cells
            r = run_cell(c, hill_k0, hill_n, tau, t_half, lam_list(j), o);
            exC(i,j) = exC(i,j) + r.exact(1);  osC(i,j) = osC(i,j) + r.sust(1);
            if i == 1, exS(j) = exS(j) + r.exact(2);  osS(j) = osS(j) + r.sust(2); end
        end
    end
end
fprintf('sweep C done\n');

%% --- (D) sparsity prior ---------------------------------------------------
P = struct('ok', [], 'exact', [], 'sust', [], 'hill', [], 'hill_sd', []);
for q = 1:numel(prior_list)
    o = rmfield(ri_opts, 'prior');  o.prior = prior_list{q};
    okq = 0;  exq = 0;  osq = 0;  hq = [];  sq = [];
    for c = 1:n_cells
        r = run_cell(c, hill_k0, hill_n, tau, t_half, lambda, o);
        okq = okq + r.ok;  exq = exq + r.exact(1);  osq = osq + r.sust(1);
        if r.ok, hq(end+1) = r.hill;  sq(end+1) = r.hill_sd; end %#ok<AGROW>
    end
    P(q) = struct('ok', okq, 'exact', exq, 'sust', osq, 'hill', hq, 'hill_sd', sq);
end

%% --- RESULTS --------------------------------------------------------------
labels = {'RI-SINDy', 'Bounded SINDy'};
show = @(name, M, rl, cl, fmt) fprintf('\n%s   (rows %s, columns %s)\n%s', name, rl{1}, cl{1}, ...
        [sprintf('%8s', ''), sprintf('%8.2f', cl{2}), newline, ...
         sprintf(['%8.2f' repmat(fmt, 1, numel(cl{2})) '\n'], [rl{2}(:), M]')]);
for m = 1:2
    show(sprintf('(A) %s: Hill share', labels{m}),        Z.A.share(:,:,m), {'n', n_list}, {'K0', k0_list}, '%8.2f');
    show(sprintf('(A) %s: cells oscillating', labels{m}), Z.A.sust(:,:,m),  {'n', n_list}, {'K0', k0_list}, '%8.0f');
    show(sprintf('(A) %s: cells {y, Hill}', labels{m}),   Z.A.exact(:,:,m), {'n', n_list}, {'K0', k0_list}, '%8.0f');
    show(sprintf('(B) %s: Hill share', labels{m}),        Z.B.share(:,:,m), {'tau (h)', tau_list}, {'half-life (min)', thalf_list}, '%8.2f');
    show(sprintf('(B) %s: cells oscillating', labels{m}), Z.B.sust(:,:,m),  {'tau (h)', tau_list}, {'half-life (min)', thalf_list}, '%8.0f');
    show(sprintf('(B) %s: cells {y, Hill}', labels{m}),   Z.B.exact(:,:,m), {'tau (h)', tau_list}, {'half-life (min)', thalf_list}, '%8.0f');
end
show('(C) RI-SINDy: cells {y, Hill}',   exC, {'eta', eta_list}, {'lambda', lam_list}, '%8.0f');
show('(C) RI-SINDy: cells oscillating', osC, {'eta', eta_list}, {'lambda', lam_list}, '%8.0f');
fprintf('\n(C) Bounded SINDy vs lambda %s: cells {y, Hill} %s, cells oscillating %s\n', ...
        mat2str(lam_list), mat2str(exS), mat2str(osS));
tot = @(M) sum(M(:));
fprintf('\nOver all %d (A) and (B) settings x %d cells: oscillating cell-settings RI %d vs SINDy %d; {y, Hill} RI %d vs SINDy %d\n', ...
        numel(Z.A.sust(:,:,1)) + numel(Z.B.sust(:,:,1)), n_cells, ...
        tot(Z.A.sust(:,:,1)) + tot(Z.B.sust(:,:,1)), tot(Z.A.sust(:,:,2)) + tot(Z.B.sust(:,:,2)), ...
        tot(Z.A.exact(:,:,1)) + tot(Z.B.exact(:,:,1)), tot(Z.A.exact(:,:,2)) + tot(Z.B.exact(:,:,2)));
fprintf('\n(D) RI-SINDy, sparsity prior (central setting)\n');
fprintf('%-10s %10s %10s %12s %20s %18s\n', 'prior', 'converged', '{y, Hill}', 'oscillating', 'Hill coef', 'Hill posterior SD');
for q = 1:numel(prior_list)
    fprintf('%-10s %8d/%d %8d/%d %10d/%d %12.2f +/- %.2f %18.3f\n', prior_list{q}, P(q).ok, n_cells, P(q).exact, n_cells, ...
            P(q).sust, n_cells, mean(P(q).hill), std(P(q).hill), mean(P(q).hill_sd));
end

%% --- FIGURES (pcolor on cell edges so every grid cell is drawn) -----------
edges = @(v) [v(1) - (v(2)-v(1))/2, (v(1:end-1) + v(2:end))/2, v(end) + (v(end)-v(end-1))/2];
pad   = @(z) [z, z(:,end); z(end,:), z(end,end)];
spec  = {'A', k0_list, n_list, 'Hill half-saturation K_0', 'Hill exponent n', 'Hill parameters K_0 and n';
         'B', thalf_list, tau_list, 'half-life prior (min)', 'delay \tau (h)', 'half-life prior and delay'};
for g = 1:2
    [XX, YY] = meshgrid(edges(spec{g,2}), edges(spec{g,3}));
    figure('Color', 'w', 'Position', [80 80 900 640]);
    for m = 1:2
        fl = {'share', 'sust'};  tl = {'Hill share of Mdm2 production', sprintf('cells (of %d) oscillating', n_cells)};
        lim = {[0 1], [0 n_cells]};
        for q = 1:2
            subplot(2, 2, (m-1)*2 + q);
            pcolor(XX, YY, pad(Z.(spec{g,1}).(fl{q})(:,:,m)));  caxis(lim{q});  colorbar;
            set(gca, 'XTick', spec{g,2}, 'YTick', spec{g,3}, 'FontSize', 8, 'Layer', 'top');
            xlabel(spec{g,4});  ylabel(spec{g,5});
            title(sprintf('%s: %s', labels{m}, tl{q}), 'FontWeight', 'normal', 'FontSize', 9);
        end
    end
    sgtitle(sprintf('p53-Mdm2: sensitivity to the %s', spec{g,6}));
end
[XX, YY] = meshgrid(exp(edges(log(lam_list))), edges(eta_list));
figure('Color', 'w', 'Position', [80 80 1250 360]);
C3 = {exC, 'RI-SINDy: cells with Mdm2 eq = \{y, Hill\}'; osC, 'RI-SINDy: cells oscillating'};
for q = 1:2
    subplot(1, 3, q);
    pcolor(XX, YY, pad(C3{q,1}));  caxis([0 n_cells]);  colorbar;  set(gca, 'XScale', 'log');
    set(gca, 'XTick', lam_list, 'YTick', eta_list, 'FontSize', 8, 'Layer', 'top');
    xlabel('threshold \lambda');  ylabel('relaxation \eta');  title(C3{q,2}, 'FontWeight', 'normal', 'FontSize', 9);
end
subplot(1, 3, 3);
semilogx(lam_list, exS, '-s', lam_list, osS, '--o', 'LineWidth', 1.2);  ylim([0 n_cells]);
xlabel('threshold \lambda');  ylabel('cells');  legend({'\{y, Hill\}', 'oscillating'}, 'Location', 'best');
title('Bounded SINDy (no \eta)', 'FontWeight', 'normal', 'FontSize', 9);  box on;
sgtitle('p53-Mdm2: sensitivity to \lambda and \eta');


%% =========================================================================
%  Fit one cell with RI-SINDy and bounded SINDy at one setting and score it
%% =========================================================================
function r = score_cell(Dc, k0, n, tu, th, lam, opts, K_check, libfun_kn, bounds, reg_col, dvar, target, dt, T_sim, osc_frac)
    t = Dc.t;  X = Dc.X;  [lb, ub] = bounds(X(:,2), th);
    Theta = libfun_kn(X(:,1), X(:,2), get_delayed(t, X(:,dvar), tu), k0, n);
    [Xi, Xi_var, ok] = fit_risindy(Theta, Dc.dXdt, lb, ub, reg_col, 'drift', opts, K_check);
    Xi(abs(Xi) < lam) = 0;
    XiS = zeros(7, 2);
    XiS(:,1) = stlsq_bounded(Theta, Dc.dXdt(:,1), [lb{1}, 0], [ub{1}, 0],   lam, []);
    XiS(:,2) = stlsq_bounded(Theta, Dc.dXdt(:,2), [lb{2}, 0], [ub{2}, inf], lam, 7);
    lib = @(a, b, d) libfun_kn(a, b, d, k0, n);
    M = {Xi, XiS};  use = [ok, true];
    r.ok = ok;  r.share = nan(1, 2);  r.exact = false(1, 2);  r.sust = false(1, 2);
    r.hill = Xi(7,2);  r.hill_sd = sqrt(Xi_var(7,2));
    for m = 1:2
        if ~use(m), continue; end
        r.share(m) = hill_share(Theta, M{m}(:,2), 7);
        r.exact(m) = isequal(M{m}(:,2)' ~= 0, target);
        s = score_model(simulate(M{m}, lib, t, X, dvar, tu, dt, T_sim, [0 0]), X, dt, osc_frac);
        r.sust(m) = s.sustained;
    end
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
