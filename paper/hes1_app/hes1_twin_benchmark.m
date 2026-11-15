%% =========================================================================
%  Hes1 synthetic benchmark -- every method against a known ground truth
%
%  Simulates a Hes1-type delayed negative-feedback model
%      dM/dt = -a*M + b / (1 + (P(t-tau)/K0)^n)
%      dP/dt =  c*M - d*P
%  samples it like the experiment (every 0.5 h for 12 h), adds measurement
%  noise (noise_level x SD of each clean series, at the sample times),
%  spline-interpolates to dt = 0.05 h and fits the nine methods of
%  hes1_comparisons.m with the same library [1, M, M^2, P, P^2, Hill],
%  bounds and settings. The decay bounds come from a half-life prior that
%  is f_prior = 0.9 of the true rate (valid but loose, as a literature
%  half-life would be). The true (K0, n, tau) are given to every method
%  except SR3 with the search on.
%
%  Part 1, noise grid x n_rep realizations, both truths. Reported per
%  method as mean +/- SD over realizations: exact structure of both
%  equations, true terms found, spurious terms, coefficient error,
%  Hill-coefficient error, system-wide relative L2 and RMSE of the
%  free-running model against the noise-free truth, mean-level error,
%  amplitude ratio, sustained oscillation. RI-SINDy also: coverage of its
%  95% credible intervals.
%  Part 2, decoy Hill term: a Hill activation of protein by delayed mRNA,
%  which is NOT in the true system, is added to the protein equation (for
%  RI-SINDy as a regulatory term set by the drift balance; for the
%  regression methods as an ordinary candidate). Reported: how often the
%  decoy survives the threshold, and its coefficient.
%
%  Requires: Optimization Toolbox (lsqlin, lsqnonlin), Signal Processing
%  Toolbox (sgolayfilt, xcorr). Runtime: about 15 minutes.
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

%% --- SETTINGS -------------------------------------------------------------
rng(42);                                % noise, bootstraps, SR3 restarts
noise_list = [0 0.02 0.05 0.10];        % x SD of each clean series
n_rep      = 20;                        % realizations per noise level (1 at 0%)
f_prior    = 0.9;                       % decay bound = f_prior x true decay rate
dt = 0.05;  dt_sample = 0.5;  T_obs = 12;  lambda = 0.05;  T_sim = 120;  osc_frac = 0.2;
ri_opts = struct('eta_pin', 0.4, 'eta_weight', 0.4, 'eta_drain', 0, ...
                 'n_iter', 60, 'w_threshold', 1e2, 'prior', 'SpikeSlab');
K_check = 100;
sr3 = struct('start', [1.5 4], 'p_lb', [0.5 1], 'p_ub', [10 20]);

% Ground truths: a = d = 1.8 /h (half-life 23 min), K0 = 2.7, tau = 0.3 h
truths = struct('n', {9, 5}, 'b', {10, 20}, 'c', {2, 4}, 'label', {'n = 9, b = 10', 'n = 5, b = 20'});
a = 1.8;  d = 1.8;  K0 = 2.7;  tau = 0.30;

reg_col   = [6, 0];  dvar = 2;
libfun_kn = @(m, p, dd, k0, n) [ones(size(m)), m, m.^2, p, p.^2, 1 ./ (1 + (max(dd, 0)/k0).^n)];
labels = {'RI-SINDy', 'RI loop, LS Hill', 'Bounded SINDy', 'Hill removed', 'Ensemble-SINDy', ...
          'Nullcline-SINDy', 'SR3, K0 n search', 'SR3, K0 n fixed', 'Traditional SINDy'};
n_m = numel(labels);  nL = numel(noise_list);  nT = numel(truths);
fields = {'exact', 'found', 'spur', 'cerr', 'herr', 'rel', 'rmseM', 'rmseP', 'lvlP', 'ampP', 'sust'};
for k = 1:numel(fields), Q.(fields{k}) = nan(nT, nL, n_m, n_rep); end
cover = nan(nT, nL, n_rep, 4);          % RI-SINDy: true value inside the 95% interval (M, Hill | M, P)

%% --- PART 1: noise grid ---------------------------------------------------
for it = 1:nT
    tr = truths(it);
    Xi_true = zeros(6, 2);  Xi_true([2 6], 1) = [-a; tr.b];  Xi_true([2 4], 2) = [tr.c; -d];
    for l = 1:nL
        for r = 1:n_rep
            if noise_list(l) == 0 && r > 1, break; end
            [tg, X, dXdt, Xc] = make_twin(a, tr.b, tr.c, d, K0, tr.n, tau, T_obs, dt_sample, dt, noise_list(l));
            lb = {[0, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf]};          % BOUNDS (as for real Hes1)
            ub = {[0, -f_prior*a, 0, 0, 0], [0, inf, inf, -f_prior*d, 0]};
            R = fit_all_methods(tg, X, dXdt, libfun_kn, dvar, tau, K0, tr.n, lb, ub, reg_col, ...
                                lambda, ri_opts, K_check, sr3, true(1, n_m));
            for m = 1:n_m
                if ~R.ok(m), continue; end
                Xi = R.Xi{m};  lib = @(u, v, w) libfun_kn(u, v, w, R.k0(m), R.n(m));
                s = score_model(simulate(Xi, lib, tg, X, dvar, tau, dt, T_sim, R.off(m,:)), Xc, dt, osc_frac);
                st = Xi_true ~= 0;  si = Xi ~= 0;
                Q.exact(it,l,m,r) = isequal(si, st);
                Q.found(it,l,m,r) = nnz(si & st);
                Q.spur(it,l,m,r)  = nnz(si & ~st);
                Q.cerr(it,l,m,r)  = 100*norm(Xi - Xi_true, 'fro') / norm(Xi_true, 'fro');
                Q.herr(it,l,m,r)  = 100*(Xi(6,1) - tr.b) / tr.b;
                Q.rel(it,l,m,r)   = s.rel_sys;  Q.rmseM(it,l,m,r) = s.rmse(1);  Q.rmseP(it,l,m,r) = s.rmse(2);
                Q.lvlP(it,l,m,r)  = s.level(2);  Q.ampP(it,l,m,r) = s.amp(2);   Q.sust(it,l,m,r) = s.sustained;
            end
            % RI-SINDy credible-interval coverage (unthresholded fit)
            Th = libfun_kn(X(:,1), X(:,2), get_delayed(tg, X(:,2), tau), K0, tr.n);
            [Xr, Vr, okr] = fit_risindy(Th, dXdt, lb, ub, reg_col, 'drift', ri_opts, 0);
            if okr
                idx = [2 1; 6 1; 2 2; 4 2];
                for q = 1:4
                    e = Xr(idx(q,1), idx(q,2)) - Xi_true(idx(q,1), idx(q,2));
                    cover(it,l,r,q) = abs(e) <= 1.96*sqrt(Vr(idx(q,1), idx(q,2)));
                end
            end
        end
        fprintf('%s, noise %.0f%% done\n', tr.label, 100*noise_list(l));
    end
end

%% --- PART 2: decoy Hill term in the protein equation ----------------------
tr = truths(1);
dec = struct('surv', nan(nL, 3, n_rep), 'coef', nan(nL, 3, n_rep), 'exact', nan(nL, 3, n_rep));
dec_labels = {'RI-SINDy', 'Bounded SINDy', 'Traditional SINDy'};
for l = 1:nL
    for r = 1:n_rep
        if noise_list(l) == 0 && r > 1, break; end
        [tg, X, dXdt] = make_twin(a, tr.b, tr.c, d, K0, tr.n, tau, T_obs, dt_sample, dt, noise_list(l));
        k0d = median(X(:,1));                                           % decoy: Hill activation of P by M(t - tau)
        Md  = max(get_delayed(tg, X(:,1), tau), 0);
        Th  = [libfun_kn(X(:,1), X(:,2), get_delayed(tg, X(:,2), tau), K0, tr.n), Md.^4 ./ (k0d^4 + Md.^4)];
        st  = false(7, 2);  st([2 6], 1) = true;  st([2 4], 2) = true;
        lb = {[0, -inf, -inf, -inf, -inf], [0, 0, 0, -inf, -inf]};
        ub = {[0, -f_prior*a, 0, 0, 0], [0, inf, inf, -f_prior*d, 0]};
        XiD = cell(1, 3);
        Xi = fit_risindy(Th, dXdt, lb, ub, [6, 7], 'drift', ri_opts, 0);  Xi(abs(Xi) < lambda) = 0;  XiD{1} = Xi;
        Xi = zeros(7, 2);
        Xi(:,1) = stlsq_bounded(Th, dXdt(:,1), [lb{1}, 0, 0], [ub{1}, inf, 0], lambda, 6);
        Xi(:,2) = stlsq_bounded(Th, dXdt(:,2), [lb{2}, 0, 0], [ub{2}, 0, inf], lambda, []);  XiD{2} = Xi;
        Xi = zeros(7, 2);
        for v = 1:2, Xi(:,v) = stlsq_bounded(Th, dXdt(:,v), -inf(1,7), inf(1,7), lambda, []); end
        XiD{3} = Xi;
        for q = 1:3
            dec.surv(l,q,r)  = XiD{q}(7,2) ~= 0;
            dec.coef(l,q,r)  = XiD{q}(7,2);
            dec.exact(l,q,r) = isequal(XiD{q} ~= 0, st);
        end
    end
end

%% --- RESULTS --------------------------------------------------------------
ms = @(v) sprintf('%.2f+/-%.2f', mean(v(:), 'omitnan'), std(v(:), 'omitnan'));
pc = @(v) sprintf('%.0f%%', 100*mean(v(:), 'omitnan'));
for it = 1:nT
    fprintf('\n=====================================================================================\n');
    fprintf(' Truth %s: dM/dt = -%.1f M + %g Hill(P(t-%.1f)),  dP/dt = %g M - %.1f P\n', truths(it).label, a, truths(it).b, tau, truths(it).c, d);
    fprintf(' %d realizations per noise level; decay prior f = %.1f; lambda = %.2f\n', n_rep, f_prior, lambda);
    fprintf('=====================================================================================\n');
    fprintf('%-18s %6s %7s %6s %11s %13s %14s %12s %12s %12s %12s %12s %7s\n', 'method', 'noise', 'exact', 'found', 'spurious', ...
            'coef err %', 'Hill err %', 'rel L2 sys', 'RMSE M', 'RMSE P', '|level P|', 'amp P', 'sust.');
    for m = 1:n_m
        for l = 1:nL
            g = @(f) squeeze(Q.(f)(it,l,m,:));
            fprintf('%-18s %5.0f%% %7s %6.2f %11s %13s %14s %12s %12s %12s %12s %12s %7s\n', labels{m}, 100*noise_list(l), ...
                    pc(g('exact')), mean(g('found'), 'omitnan'), ms(g('spur')), ms(g('cerr')), ms(g('herr')), ...
                    ms(g('rel')), ms(g('rmseM')), ms(g('rmseP')), ms(abs(g('lvlP'))), ms(g('ampP')), pc(g('sust')));
        end
    end
    fprintf('(exact: both equations have exactly the true terms; found: of %d true terms; rel L2 sys: both\n', 4);
    fprintf(' variables stacked, free-running model vs noise-free truth on 0-%g h; Hill err: signed, %% of b)\n', T_obs);
    fprintf('\nRI-SINDy 95%% credible-interval coverage (true value inside), by noise level:\n');
    fprintf('%8s %10s %10s %10s %10s\n', 'noise', 'mRNA: M', 'mRNA: Hill', 'prot: M', 'prot: P');
    for l = 1:nL
        fprintf('%7.0f%% %10s %10s %10s %10s\n', 100*noise_list(l), pc(cover(it,l,:,1)), pc(cover(it,l,:,2)), ...
                pc(cover(it,l,:,3)), pc(cover(it,l,:,4)));
    end
end

fprintf('\n=== Decoy Hill activation of protein by M(t - tau), truth %s ===\n', truths(1).label);
fprintf('%-18s %6s %16s %18s %16s\n', 'method', 'noise', 'decoy survives', 'decoy coefficient', 'exact structure');
for q = 1:3
    for l = 1:nL
        fprintf('%-18s %5.0f%% %16s %18s %16s\n', dec_labels{q}, 100*noise_list(l), pc(dec.surv(l,q,:)), ...
                ms(dec.coef(l,q,:)), pc(dec.exact(l,q,:)));
    end
end

%% --- FIGURE ---------------------------------------------------------------
show = [1 3 5 6 7 8 9];  mk = {'-o', '-^', '-.d', '--v', ':>', '-<', '--p'};
pn = {'exact', 'exact structure, both equations (%)', 100, [0 100];
      'cerr',  'coefficient error (%)', 1, [0 100];
      'rel',   'relative L2 error, system-wide', 1, [0 1.2];
      'ampP',  'protein amplitude ratio (model / truth)', 1, [0 1.5]};
for it = 1:nT
    figure('Color', 'w', 'Position', [60 60 1300 380]);
    for q = 1:4
        subplot(1, 4, q);  hold on;
        for j = 1:numel(show)
            v = squeeze(mean(Q.(pn{q,1})(it,:,show(j),:), 4, 'omitnan')) * pn{q,3};
            plot(100*noise_list, v, mk{j}, 'LineWidth', 1.1);
        end
        xlabel('noise (% of SD)');  ylabel(pn{q,2});  ylim(pn{q,4});  box on;
        if q == 1, legend(labels(show), 'FontSize', 6, 'Location', 'best'); end
    end
    sgtitle(sprintf('Synthetic Hes1 (%s): mean over %d realizations', truths(it).label, n_rep));
end


%% =========================================================================
%  Synthetic data: simulate the truth, sample, add noise, interpolate
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
