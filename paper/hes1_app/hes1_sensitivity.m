%% =========================================================================
%  Hes1 sensitivity -- every assumed setting, against a known ground truth
%
%  Uses the synthetic Hes1 model and sampling of hes1_twin_benchmark.m
%  (truth n = 9, b = 10, c = 2, a = d = 1.8, K0 = 2.7, tau = 0.3) so that
%  term recovery can be scored. RI-SINDy and bounded SINDy, same library,
%  bounds and settings as the real-data scripts. Swept:
%    (A) assumed Hill parameters K0 x n (truth 2.7, 9), 5% noise
%    (B) threshold lambda x relaxation factor eta (eta_pin, Eq. 28), 5% noise
%    (C) threshold lambda x decay-prior factor f (bound = f x true rate;
%        f < 1 valid but loose, f > 1 too strict), 10% noise, both truths
%    (D) sparsity prior: Spike-and-Slab, Laplace, Horseshoe, 5% noise
%  The same noise realizations are reused across the settings of a sweep.
%  Scored: exact structure of the mRNA equation {M, Hill} and of both
%  equations, Hill-coefficient error, system-wide relative L2 error of the
%  free-running model against the noise-free truth.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr). Runtime: about 15 minutes.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

%% --- SETTINGS -------------------------------------------------------------
rng(42);                                % noise realizations
n_rep = 10;  n_rep_C = 20;
dt = 0.05;  dt_sample = 0.5;  T_obs = 12;  lambda = 0.05;  f_prior = 0.9;  T_sim = 120;  osc_frac = 0.2;
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;
a = 1.8;  d = 1.8;  K0 = 2.7;  tau = 0.30;
truths = struct('n', {9, 5}, 'b', {10, 20}, 'c', {2, 4}, 'label', {'n = 9, b = 10', 'n = 5, b = 20'});

k0_list    = [1.5 2 2.7 3.5 4.5 6];          % (A) assumed K0
n_list     = [2 4 6 9 12 16];                % (A) assumed n
lam_list   = [0.01 0.02 0.05 0.1 0.2 0.5];   % (B) threshold, log-spaced
eta_list   = [0.1 0.2 0.4 0.6 0.8 1.0];      % (B) relaxation factor
lamC_list  = [0.02 0.05 0.1 0.2 0.5];        % (C) threshold
f_list     = [0.8 0.9 0.96 1.0 1.1];         % (C) decay-prior factor
prior_list = {'SpikeSlab', 'Laplace', 'Horseshoe'};   % (D)

reg_col   = [6, 0];  dvar = 2;
libfun_kn = @(m, p, dd, k0, n) [ones(size(m)), m, m.^2, p, p.^2, 1 ./ (1 + (max(dd, 0)/k0).^n)];
bounds    = @(f) deal({[0, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf]}, ...
                      {[0, -f*a, 0, 0, 0], [0, inf, inf, -f*d, 0]});
labels = {'RI-SINDy', 'Bounded SINDy'};

%% --- DATA: the same realizations for every setting of a sweep -------------
mk_set = @(tr, noise, nr) arrayfun(@(r) wrap_twin(a, tr.b, tr.c, d, K0, tr.n, tau, T_obs, dt_sample, dt, noise), ...
                                    1:nr, 'UniformOutput', false);
D5  = mk_set(truths(1), 0.05, n_rep);
D10 = {mk_set(truths(1), 0.10, n_rep_C), mk_set(truths(2), 0.10, n_rep_C)};
true_of = @(tr) [0 0; -a tr.c; 0 0; 0 -d; 0 0; tr.b 0];

%% --- (A) assumed K0 x n ---------------------------------------------------
[lb, ub] = bounds(f_prior);
A = struct('ex', zeros(numel(n_list), numel(k0_list), 2), 'rel', zeros(numel(n_list), numel(k0_list), 2));
for i = 1:numel(n_list)
    for j = 1:numel(k0_list)
        for r = 1:n_rep
            s = fit_score(D5{r}, k0_list(j), n_list(i), lambda, ri_opts, K_check, lb, ub, libfun_kn, reg_col, ...
                          dvar, tau, dt, T_sim, osc_frac, true_of(truths(1)));
            A.ex(i,j,:)  = A.ex(i,j,:)  + reshape(s.ex_m, 1, 1, 2) / n_rep;
            A.rel(i,j,:) = A.rel(i,j,:) + reshape(s.rel, 1, 1, 2) / n_rep;
        end
    end
end
fprintf('sweep A done\n');

%% --- (B) lambda x eta -----------------------------------------------------
B = struct('ex', zeros(numel(eta_list), numel(lam_list)), 'rel', zeros(numel(eta_list), numel(lam_list)), ...
           'exS', zeros(1, numel(lam_list)), 'relS', zeros(1, numel(lam_list)));
for i = 1:numel(eta_list)
    o = ri_opts;  o.eta_pin = eta_list(i);
    for j = 1:numel(lam_list)
        for r = 1:n_rep
            s = fit_score(D5{r}, K0, truths(1).n, lam_list(j), o, K_check, lb, ub, libfun_kn, reg_col, ...
                          dvar, tau, dt, T_sim, osc_frac, true_of(truths(1)));
            B.ex(i,j)  = B.ex(i,j)  + s.ex_both(1) / n_rep;
            B.rel(i,j) = B.rel(i,j) + s.rel(1) / n_rep;
            if i == 1, B.exS(j) = B.exS(j) + s.ex_both(2) / n_rep;  B.relS(j) = B.relS(j) + s.rel(2) / n_rep; end
        end
    end
end
fprintf('sweep B done\n');

%% --- (C) lambda x decay-prior factor f, 10% noise, both truths -------------
C = struct('ex', zeros(numel(f_list), numel(lamC_list), 2, 2), 'herr', zeros(numel(f_list), numel(lamC_list), 2, 2));
for it = 1:2
    for i = 1:numel(f_list)
        [lbf, ubf] = bounds(f_list(i));
        for j = 1:numel(lamC_list)
            for r = 1:n_rep_C
                s = fit_score(D10{it}{r}, K0, truths(it).n, lamC_list(j), ri_opts, K_check, lbf, ubf, libfun_kn, ...
                              reg_col, dvar, tau, dt, 0, osc_frac, true_of(truths(it)));
                C.ex(i,j,:,it)   = C.ex(i,j,:,it)   + reshape(s.ex_m, 1, 1, 2) / n_rep_C;
                C.herr(i,j,:,it) = C.herr(i,j,:,it) + reshape(abs(s.herr), 1, 1, 2) / n_rep_C;
            end
        end
    end
end
fprintf('sweep C done\n');

%% --- (D) sparsity prior ---------------------------------------------------
Dp = zeros(numel(prior_list), 4);       % exact both, exact mRNA, |Hill err| %, rel L2
for q = 1:numel(prior_list)
    o = rmfield(ri_opts, 'prior');  o.prior = prior_list{q};
    for r = 1:n_rep
        s = fit_score(D5{r}, K0, truths(1).n, lambda, o, K_check, lb, ub, libfun_kn, reg_col, ...
                      dvar, tau, dt, T_sim, osc_frac, true_of(truths(1)));
        Dp(q,:) = Dp(q,:) + [s.ex_both(1), s.ex_m(1), abs(s.herr(1)), s.rel(1)] / n_rep;
    end
end

%% --- RESULTS --------------------------------------------------------------
show = @(name, Mx, rl, cl, fmt) fprintf('\n%s   (rows %s, columns %s)\n%s', name, rl{1}, cl{1}, ...
        [sprintf('%8s', ''), sprintf('%8.2f', cl{2}), newline, ...
         sprintf(['%8.2f' repmat(fmt, 1, numel(cl{2})) '\n'], [rl{2}(:), Mx]')]);
for m = 1:2
    show(sprintf('(A) %s: mRNA eq exactly {M, Hill} (%%)', labels{m}), 100*A.ex(:,:,m), {'n', n_list}, {'K0', k0_list}, '%8.0f');
    show(sprintf('(A) %s: rel L2 vs truth', labels{m}), A.rel(:,:,m), {'n', n_list}, {'K0', k0_list}, '%8.2f');
end
show('(B) RI-SINDy: both equations exact (%)', 100*B.ex, {'eta', eta_list}, {'lambda', lam_list}, '%8.0f');
show('(B) RI-SINDy: rel L2 vs truth', B.rel, {'eta', eta_list}, {'lambda', lam_list}, '%8.2f');
fprintf('\n(B) Bounded SINDy vs lambda %s: both exact (%%) %s, rel L2 %s\n', mat2str(lam_list), ...
        mat2str(round(100*B.exS)), mat2str(B.relS, 2));
for it = 1:2
    for m = 1:2
        show(sprintf('(C) %s, %s: mRNA eq exactly {M, Hill} (%%), 10%% noise', labels{m}, truths(it).label), ...
             100*C.ex(:,:,m,it), {'f', f_list}, {'lambda', lamC_list}, '%8.0f');
        show(sprintf('(C) %s, %s: |Hill coefficient error| (%%)', labels{m}, truths(it).label), ...
             C.herr(:,:,m,it), {'f', f_list}, {'lambda', lamC_list}, '%8.1f');
    end
end
fprintf('\n(D) RI-SINDy, sparsity prior, 5%% noise, %d realizations\n', n_rep);
fprintf('%-10s %14s %14s %16s %10s\n', 'prior', 'both exact', 'mRNA exact', '|Hill err| (%)', 'rel L2');
for q = 1:numel(prior_list)
    fprintf('%-10s %13.0f%% %13.0f%% %16.1f %10.3f\n', prior_list{q}, 100*Dp(q,1), 100*Dp(q,2), Dp(q,3), Dp(q,4));
end

%% --- FIGURES (pcolor on cell edges so every grid cell is drawn) -----------
edges = @(v) [v(1) - (v(2)-v(1))/2, (v(1:end-1) + v(2:end))/2, v(end) + (v(end)-v(end-1))/2];
ledges = @(v) exp(edges(log(v)));
pad = @(z) [z, z(:,end); z(end,:), z(end,end)];
[XX, YY] = meshgrid(edges(k0_list), edges(n_list));
figure('Color', 'w', 'Position', [80 80 900 640]);
for m = 1:2
    subplot(2, 2, m);
    pcolor(XX, YY, pad(100*A.ex(:,:,m)));  caxis([0 100]);  colorbar;  hold on;  plot(K0, 9, 'w*', 'MarkerSize', 10);
    set(gca, 'XTick', k0_list, 'YTick', n_list, 'FontSize', 8, 'Layer', 'top');
    xlabel('assumed K_0');  ylabel('assumed n');  title(sprintf('%s: mRNA eq exact (%%)', labels{m}), 'FontWeight', 'normal');
    subplot(2, 2, 2 + m);
    pcolor(XX, YY, pad(A.rel(:,:,m)));  caxis([0 1]);  colorbar;  hold on;  plot(K0, 9, 'w*', 'MarkerSize', 10);
    set(gca, 'XTick', k0_list, 'YTick', n_list, 'FontSize', 8, 'Layer', 'top');
    xlabel('assumed K_0');  ylabel('assumed n');  title(sprintf('%s: rel L2 vs truth', labels{m}), 'FontWeight', 'normal');
end
sgtitle('Synthetic Hes1: wrong Hill parameters (star: truth), 5% noise');

[XX, YY] = meshgrid(ledges(lam_list), edges(eta_list));
figure('Color', 'w', 'Position', [80 80 1250 360]);
subplot(1, 3, 1);  pcolor(XX, YY, pad(100*B.ex));  caxis([0 100]);  colorbar;  set(gca, 'XScale', 'log');
set(gca, 'XTick', lam_list, 'YTick', eta_list, 'FontSize', 8, 'Layer', 'top');
xlabel('\lambda');  ylabel('\eta');  title('RI-SINDy: both equations exact (%)', 'FontWeight', 'normal');
subplot(1, 3, 2);  pcolor(XX, YY, pad(B.rel));  caxis([0 1]);  colorbar;  set(gca, 'XScale', 'log');
set(gca, 'XTick', lam_list, 'YTick', eta_list, 'FontSize', 8, 'Layer', 'top');
xlabel('\lambda');  ylabel('\eta');  title('RI-SINDy: rel L2 vs truth', 'FontWeight', 'normal');
subplot(1, 3, 3);  yyaxis left;  semilogx(lam_list, 100*B.exS, '-s', 'LineWidth', 1.2);  ylabel('both exact (%)');  ylim([0 100]);
yyaxis right;  semilogx(lam_list, B.relS, '--o', 'LineWidth', 1.2);  ylabel('rel L2');
xlabel('\lambda');  title('Bounded SINDy (no \eta)', 'FontWeight', 'normal');  box on;
sgtitle('Synthetic Hes1: threshold and relaxation factor, 5% noise');

[XX, YY] = meshgrid(ledges(lamC_list), edges(f_list));
figure('Color', 'w', 'Position', [80 80 900 640]);
for it = 1:2
    for m = 1:2
        subplot(2, 2, (it-1)*2 + m);
        pcolor(XX, YY, pad(100*C.ex(:,:,m,it)));  caxis([0 100]);  colorbar;  set(gca, 'XScale', 'log');
        set(gca, 'XTick', lamC_list, 'YTick', f_list, 'FontSize', 8, 'Layer', 'top');
        xlabel('\lambda');  ylabel('decay-prior factor f');
        title(sprintf('%s, %s: mRNA eq exact (%%)', labels{m}, truths(it).label), 'FontWeight', 'normal', 'FontSize', 9);
    end
end
sgtitle('Synthetic Hes1: threshold and decay prior, 10% noise');


%% =========================================================================
%  Fit one data set with RI-SINDy and bounded SINDy and score against truth
%% =========================================================================
function s = fit_score(Dr, k0, n, lam, opts, K_check, lb, ub, libfun_kn, reg_col, dvar, tau, dt, T_sim, osc_frac, Xi_true)
    Th = libfun_kn(Dr.X(:,1), Dr.X(:,2), get_delayed(Dr.t, Dr.X(:,2), tau), k0, n);
    [Xr, ~, ok] = fit_risindy(Th, Dr.dXdt, lb, ub, reg_col, 'drift', opts, K_check);
    Xr(abs(Xr) < lam) = 0;
    XS = zeros(6, 2);
    XS(:,1) = stlsq_bounded(Th, Dr.dXdt(:,1), [lb{1}, 0], [ub{1}, inf], lam, 6);
    XS(:,2) = stlsq_bounded(Th, Dr.dXdt(:,2), [lb{2}, 0], [ub{2}, 0],   lam, []);
    M = {Xr, XS};  okm = [ok, true];  st = Xi_true ~= 0;
    s.ex_m = zeros(1, 2);  s.ex_both = zeros(1, 2);  s.herr = nan(1, 2);  s.rel = nan(1, 2);
    lib = @(u, v, w) libfun_kn(u, v, w, k0, n);
    for m = 1:2
        if ~okm(m), continue; end
        s.ex_m(m)    = isequal(M{m}(:,1) ~= 0, st(:,1));
        s.ex_both(m) = isequal(M{m} ~= 0, st);
        s.herr(m)    = 100*(M{m}(6,1) - Xi_true(6,1)) / Xi_true(6,1);
        if T_sim > 0
            sc = score_model(simulate(M{m}, lib, Dr.t, Dr.X, dvar, tau, dt, T_sim, [0 0]), Dr.Xc, dt, osc_frac);
            s.rel(m) = min(sc.rel_sys, 2);             % a diverged model scores 2
            if isnan(sc.rel_sys), s.rel(m) = 2; end
        end
    end
end

function Dr = wrap_twin(a, b, c, d, K0, n, tau, T, dt_sample, dt, noise)
    [Dr.t, Dr.X, Dr.dXdt, Dr.Xc] = make_twin(a, b, c, d, K0, n, tau, T, dt_sample, dt, noise);
end

%% =========================================================================
%  Synthetic data: simulate the truth, sample, add noise, interpolate
%  (identical to hes1_twin_benchmark.m)
%% =========================================================================
function [tg, X, dXdt, Xc] = make_twin(a, b, c, d, K0, n, tau, T, dt_sample, dt, noise)
    h = 0.0025;  N = round(T/h) + 1;  t = (0:N-1)'*h;  L = round(tau/h);   % fine Euler step, constant history
    M = zeros(N,1);  P = zeros(N,1);  M(1) = 0.7;  P(1) = 1.0;
    for k = 2:N
        if k-1-L >= 1, Pd = P(k-1-L); else, Pd = P(1); end
        M(k) = M(k-1) + h*(-a*M(k-1) + b/(1 + (Pd/K0)^n));
        P(k) = P(k-1) + h*( c*M(k-1) - d*P(k-1));
    end
    ts = (0:dt_sample:T)';  tg = (0:dt:T)';
    Ms = interp1(t, M, ts);  Ps = interp1(t, P, ts);
    X  = [interp1(ts, Ms + noise*std(Ms)*randn(size(Ms)), tg, 'spline'), ...
          interp1(ts, Ps + noise*std(Ps)*randn(size(Ps)), tg, 'spline')];
    Xc = [interp1(t, M, tg), interp1(t, P, tg)];
    dXdt = [smooth_derivative(X(:,1), dt, 3, 11), smooth_derivative(X(:,2), dt, 3, 11)];
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
