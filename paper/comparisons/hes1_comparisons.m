%% =========================================================================
%  Hes1 mRNA-Protein Network
%  SR3 (parametric-library, Champion et al. 2020)  vs
%  Nullcline-Reconstruction SINDy (Prokop, Frolov & Gelens 2024)  vs
%  Traditional SINDy (plain STLSQ, naive baseline)
%
%  RI-SINDy is NOT re-fit here -- its published coefficients are
%  hardcoded below and only forward-integrated, so all four methods can
%  be compared side by side on identical footing.
%
%  Requires: Optimization Toolbox (lsqnonlin, fminsearch)
%            Signal Processing Toolbox (sgolayfilt)
%
%  SCOPE NOTE: this script fits and evaluates only on the available
%  interpolated window (t = 0-15 hrs, see the "Hes1 Data" folder at the
%  repo root). The original version of this comparison additionally
%  forward-integrated out to t = 35 hrs to visualize extrapolation
%  behavior, using an extrapolated data file not currently in this
%  repository. Add it to "Hes1 Data" and extend T_END below to restore that.
%
%  ----------------------------------------------------------------------
%  FAIRNESS NOTE:
%
%  Nullcline-SINDy is given the TRUE Hill hyperparameters (hill_k0,
%  hill_n), exactly as RI-SINDy is. It has no mechanism of its own for
%  identifying an embedded nonlinear parameter -- its only contribution
%  is correcting the phase-space position of the data before fitting,
%  not the shape of the regulatory function itself.
%
%  SR3 is deliberately NOT given the true hill_k0/hill_n. Its entire
%  reason for being tested is that it claims to jointly discover an
%  embedded nonlinear library parameter alongside the linear
%  coefficients -- handing it the true values would trivialize it into
%  ordinary thresholded least squares with a known column. SR3 instead
%  starts from a generic, uninformed initial guess for (K0, n) and has
%  to find them itself.
%
%  Traditional SINDy is the naive baseline: true K0/n are given (like
%  Nullcline-SINDy), but NO fairness treatment is applied at all -- the
%  Hill column competes as an ordinary candidate in BOTH equations and
%  is thresholded like any other term, unlike SR3/Nullcline-SINDy/
%  RI-SINDy, which all structurally exclude the Hill column from the
%  protein equation (RI-SINDy hardcodes that exclusion; SR3 and
%  Nullcline-SINDy mirror it manually below for a fair comparison).
%  ----------------------------------------------------------------------
% =========================================================================

close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, 'utils')));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

% Dock every figure into a single tabbed window instead of opening
% separate pop-up windows. To go back to normal floating windows, run
% set(0,'DefaultFigureWindowStyle','normal') in the command line.
set(0,'DefaultFigureWindowStyle','docked');

%% --- SHARED HYPERPARAMETERS (matching the paper/hes1 RI-SINDy run) ------
polyorder    = 2;
dt           = 0.05;
tau          = 0.25;
hill_n_true  = 9;
hill_k0_true = 2.7;
hill_func    = @(p,k,n) 1./(1+(p./k).^n);

%% --- LOAD DATA ------------------------------------------------------------
dataDir = fullfile(thisDir, '..', '..', 'Hes1 Data');

x_data_all = readmatrix(fullfile(dataDir, 'interpmRNAData.csv'));
y_data_all = readmatrix(fullfile(dataDir, 'interpHes1Data.csv'));
mRNA       = readmatrix(fullfile(dataDir, 'mRNA.csv'));
mRNAtime   = readmatrix(fullfile(dataDir, 'mRNAtime.csv'));
hes1       = readmatrix(fullfile(dataDir, 'hes1.csv'));
hes1time   = readmatrix(fullfile(dataDir, 'hes1time.csv'));

t_all = linspace(0, 15, numel(x_data_all))';
t     = t_all;   % fit/evaluate over the same window (see SCOPE NOTE above)

N = round(0.9 * numel(x_data_all));   % training window; rest held out for error metrics
x_tr = x_data_all(1:N);
y_tr = y_data_all(1:N);

y_tau_all = get_delayed(t_all, y_data_all, tau);
y_tau_tr  = y_tau_all(1:N);

%% --- NUMERICAL DERIVATIVES -----------------------------------------------
sgolay_p = 3; sgolay_f = 11;
dxdt_tr = smooth_derivative(x_tr, dt, sgolay_p, sgolay_f);
dydt_tr = smooth_derivative(y_tr, dt, sgolay_p, sgolay_f);
dXdt_tr = [dxdt_tr, dydt_tr];

fprintf('=========================================================\n');
fprintf(' Hes1: SR3 vs Nullcline-SINDy vs Traditional SINDy\n');
fprintf('=========================================================\n');
fprintf('N = %d training points (of %d total), dt = %.3f\n', N, numel(x_data_all), dt);

%% --- Library collinearity heatmap, on the same data every method fits on ---
Theta_full_tr = [build_poly_library(x_tr, y_tr, polyorder), hill_func(y_tau_tr, hill_k0_true, hill_n_true)];
libNames = {'1','x','x^2','y','y^2','Hill(y_tau)'};
plot_collinearity_heatmap(Theta_full_tr(:,2:end), libNames(2:end), 'hes1');

%% =========================================================================
%  SR3 -- parametric-library relax-and-split (Champion et al. 2020)
%% =========================================================================
fprintf('\n--- Fitting SR3 (jointly discovering K0 and n from scratch) ---\n');
tic;
[Xi_sr3, k0_sr3, n_sr3] = run_SR3_hes1(x_tr, y_tr, y_tau_tr, dXdt_tr, polyorder);
fprintf('SR3 finished in %.1f s\n', toc);
fprintf('SR3 discovered Hill hyperparameters: K0 = %.4f, n = %.4f\n', k0_sr3, n_sr3);
fprintf('(True values used elsewhere in this work: K0 = %.2f, n = %d)\n', hill_k0_true, hill_n_true);
print_coeffs('SR3', Xi_sr3, libNames, {'mRNA','Protein'});

rhs_sr3 = @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,k0_sr3,n_sr3)] * Xi_sr3;
Xhat_sr3 = forward_integrate_hes1(rhs_sr3, t, t_all, x_data_all(1), y_data_all(1), y_data_all, tau, dt);

%% =========================================================================
%  NULLCLINE-RECONSTRUCTION SINDy (Prokop, Frolov & Gelens 2024, adapted)
%% =========================================================================
fprintf('\n--- Fitting Nullcline-Reconstruction SINDy (true K0, n given) ---\n');
tic;
[Xi_null, off_best, score_best] = run_NullclineSINDy_hes1( ...
    x_tr, y_tr, y_tau_tr, dXdt_tr, polyorder, hill_k0_true, hill_n_true);
fprintf('Nullcline-SINDy finished in %.1f s\n', toc);
fprintf('Best offset: [x: %.4f, y: %.4f]   score = %.4f\n', off_best(1), off_best(2), score_best);
print_coeffs('Nullcline-SINDy', Xi_null, libNames, {'mRNA','Protein'});

rhs_null = @(xp,yp,ytk) [poly_library_row(xp+off_best(1), yp+off_best(2), polyorder), ...
                          hill_func(ytk+off_best(2), hill_k0_true, hill_n_true)] * Xi_null;
Xhat_null = forward_integrate_hes1(rhs_null, t, t_all, x_data_all(1), y_data_all(1), y_data_all, tau, dt);

%% =========================================================================
%  TRADITIONAL SINDy -- plain STLSQ, true K0/n given, no fairness
%  treatment at all: the Hill column competes equally in BOTH equations.
%% =========================================================================
fprintf('\n--- Fitting Traditional SINDy (plain STLSQ, true K0, n given) ---\n');
tic;
Xi_trad = run_TraditionalSINDy_hes1(x_tr, y_tr, y_tau_tr, dXdt_tr, polyorder, hill_k0_true, hill_n_true);
fprintf('Traditional SINDy finished in %.1f s\n', toc);
print_coeffs('Traditional SINDy', Xi_trad, libNames, {'mRNA','Protein'});

rhs_trad = @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,hill_k0_true,hill_n_true)] * Xi_trad;
Xhat_trad = forward_integrate_hes1(rhs_trad, t, t_all, x_data_all(1), y_data_all(1), y_data_all, tau, dt);

%% =========================================================================
%  ALREADY-IDENTIFIED RI-SINDy MODEL (hardcoded, not re-fit here)
% =========================================================================
fprintf('\n--- Propagating the already-identified RI-SINDy model ---\n');

% rows: 1, M, M^2, P, P^2, HillDelay  |  columns: mRNA, Protein
Xi_ri = [   0        ,   0       ; ...   % 1
           -1.90514   ,   0       ; ...   % M
            0         ,   0.64064 ; ...   % M^2
            0         ,  -1.01964 ; ...   % P
            0         ,   0       ; ...   % P^2
           11.76276    ,   0       ];      % HillDelay

% Posterior variance, physical units, same 6x2 layout as Xi_ri.
Xi_ri_var = [ 0.0019, 0.0032 ; ...   % 1
              0.0130, 0.0005 ; ...   % M
              0.0000, 0.0018 ; ...   % M^2
              0.0001, 0.0074 ; ...   % P
              0.0000, 0.0000 ; ...   % P^2
              0.0001, 0      ];      % HillDelay

print_coeffs('RI-SINDy', Xi_ri, libNames, {'mRNA','Protein'});

rhs_ri = @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,hill_k0_true,hill_n_true)] * Xi_ri;
Xhat_ri = forward_integrate_hes1(rhs_ri, t, t_all, x_data_all(1), y_data_all(1), y_data_all, tau, dt);

fprintf('\n--- Propagating RI-SINDy UQ band (n=100 Monte Carlo draws) ---\n');
draw_fn = @(Xi_s) forward_integrate_hes1( ...
    @(xp,yp,ytk) [poly_library_row(xp,yp,polyorder), hill_func(ytk,hill_k0_true,hill_n_true)] * Xi_s, ...
    t, t_all, x_data_all(1), y_data_all(1), y_data_all, tau, dt);
[Xenv_lo, Xenv_hi] = mc_uq_propagate(draw_fn, Xi_ri, Xi_ri_var, 100);

%% --- Error metrics ---------------------------------------------------------
fprintf('\n--- Error metrics vs. interpolated reference trajectory ---\n');
stateNames = {'mRNA','Protein'};
Xtrue = [x_data_all, y_data_all];
print_error_block('RI-SINDy',          Xtrue, Xhat_ri,   stateNames);
print_error_block('SR3',               Xtrue, Xhat_sr3,  stateNames);
print_error_block('Nullcline-SINDy',   Xtrue, Xhat_null, stateNames);
print_error_block('Traditional SINDy', Xtrue, Xhat_trad, stateNames);

%% --- Method-by-state comparison grid (rows = mRNA/Protein, cols = methods) ---
methodNames  = {'RI-SINDy','SR3','Nullcline-SINDy','Traditional SINDy'};
methodTrajs  = {Xhat_ri, Xhat_sr3, Xhat_null, Xhat_trad};
methodColors = {[0 0.2 0.8], [0.75 0 0.55], [0.85 0.25 0], [0.4 0.4 0.4]};
obsTime = {mRNAtime, hes1time};
obsVal  = {mRNA, hes1};

figure('Name','Hes1: method comparison grid','Color','w');
for d = 1:2
    for m = 1:numel(methodNames)
        subplot(2, numel(methodNames), (d-1)*numel(methodNames) + m); hold on;

        hUQ = [];
        if m == 1
            hUQ = fill([t_all; flipud(t_all)], [Xenv_hi(:,d); flipud(Xenv_lo(:,d))], ...
                methodColors{1}, 'FaceAlpha', 0.18, 'EdgeColor', 'none');
        end
        hTrue = plot(t_all, Xtrue(:,d), 'k-', 'LineWidth', 1);
        hObs  = plot(obsTime{d}, obsVal{d}, '.', 'Color', [0.35 0.35 0.35], ...
                     'MarkerSize', 5, 'LineWidth', 0.7);

        diverged = any(~isfinite(methodTrajs{m}(:,d)));
        hModel = [];
        if ~diverged
            hModel = plot(t_all, methodTrajs{m}(:,d), '--', 'Color', methodColors{m}, 'LineWidth', 1.1);
        end
        hSplit = xline(t(N), 'k:', 'LineWidth', 0.6);

        if d == 1
            title(methodNames{m}, 'FontSize', 9, 'FontWeight','normal');
        end
        if m == 1
            ylabel(stateNames{d}, 'FontSize', 8);
        end
        if d == 2
            xlabel('time', 'FontSize', 8);
        end
        if diverged
            text(mean(xlim), mean(ylim), 'model diverged', 'HorizontalAlignment','center', ...
                'FontSize', 8, 'FontAngle','italic', 'Color',[0.45 0.45 0.45]);
        end
        set(gca, 'FontSize', 7, 'Box','on');

        if d == 2 && m == 1
            legHandles = [hTrue, hObs, hModel, hUQ, hSplit];
            legLabels  = {'Interpolated data','Observed data','RI-SINDy','90% UQ band','train/test split'};
            if isempty(hModel)
                legHandles(3) = []; legLabels(3) = [];
            end
            legend(legHandles, legLabels, 'Location','best', 'FontSize',6, 'Box','on');
        end
    end
end
sgtitle('Hes1', 'FontSize', 11, 'FontWeight','normal');


%% =========================================================================
%  SR3: relax-and-split with a parametric Hill column (K0, n jointly
%  optimized alongside the linear coefficients), following Champion,
%  Zheng, Aravkin, Brunton & Kutz (2020).
%% =========================================================================
function [Xi, K0_final, n_final] = run_SR3_hes1(x, y, y_tau, dXdt, polyorder)

    nu     = 0.05;   % relaxation strength linking Xi and the sparse copy W
    lambda = 0.05;   % sparsity threshold (in normalized coefficient space)
    K_outer = 15;

    Theta_poly = build_poly_library(x, y, polyorder);   % fixed columns: 1,x,x^2,y,y^2
    p_poly = size(Theta_poly,2);

    % generic, uninformed starting guess -- not the true K0, n
    K0 = 1.5; nHill = 4;

    Xi = zeros(p_poly+1, 2);
    W  = Xi;

    optsLSQ = optimoptions('lsqnonlin','Display','off','MaxIterations',60);

    for outer = 1:K_outer
        Hcol = hill_func_local(y_tau, K0, nHill);
        Theta = [Theta_poly, Hcol];

        colscale = colscale_floor(vecnorm(Theta,2,1));
        ThetaN = Theta ./ colscale;
        targetScale = vecnorm(dXdt,2,1); targetScale(targetScale==0) = 1;
        dXdtN = dXdt ./ targetScale;

        % --- (1) Xi update: ridge pull toward the sparse copy W ---
        % Equation 2 (protein) never sees the Hill column at all, matching
        % the structural prior already built into RI-SINDy, which hardcodes
        % the protein equation's Hill coefficient to exactly zero rather
        % than thresholding it out after the fact.
        WN = (W .* colscale') ./ targetScale;
        for eq = 1:2
            if eq == 1
                idx = 1:size(ThetaN,2);
            else
                idx = 1:size(ThetaN,2)-1;   % drop the Hill column for protein
            end
            ThetaN_eq = ThetaN(:,idx);
            A_eq = ThetaN_eq'*ThetaN_eq + eye(numel(idx))/nu;
            b_eq = ThetaN_eq'*dXdtN(:,eq) + WN(idx,eq)/nu;
            xiN_eq = A_eq\b_eq;
            xiN_full = zeros(size(ThetaN,2),1);
            xiN_full(idx) = xiN_eq;
            Xi(:,eq) = (xiN_full .* targetScale(eq)) ./ colscale';
        end

        % --- (2) W update: hard threshold (sparsity proximal step) ---
        thresh = sqrt(2*lambda*nu) * (targetScale ./ mean(colscale));
        W = Xi;
        for eq = 1:2
            small = abs(W(:,eq)) < thresh(eq);
            W(small, eq) = 0;
        end

        % --- (3) (K0, n) update: nonlinear least squares, Xi held fixed ---
        objfun = @(p) sr3_hill_residual(p, y_tau, dXdt, Theta_poly, Xi);
        bestCost = inf; bestP = [K0, nHill];
        for trial = 1:3
            p0 = [1 + 8*rand(), 1 + 18*rand()];
            try
                [psol, cost] = lsqnonlin(objfun, p0, [0.5 1], [10 20], optsLSQ);
                if cost < bestCost
                    bestCost = cost; bestP = psol;
                end
            catch
                continue;
            end
        end
        K0 = bestP(1); nHill = bestP(2);
    end

    K0_final = K0;
    n_final  = nHill;
end

function res = sr3_hill_residual(p, y_tau, dXdt, Theta_poly, Xi)
    K0 = p(1); nHill = p(2);
    Hcol = hill_func_local(y_tau, K0, nHill);
    Theta = [Theta_poly, Hcol];
    res = Theta * Xi - dXdt;
    res = res(:);

    % Prevents the optimizer from settling on a (K0, n) pair that drives
    % the Hill column to near-zero norm, which would otherwise blow up
    % the recovered physical coefficient via division by a near-zero
    % norm in the Xi update.
    penaltyWeight = 5;
    degPenalty = penaltyWeight / max(norm(Hcol), 1e-6);
    res = [res; degPenalty];
end

function h = hill_func_local(p,k,n)
    h = 1./(1+(p./k).^n);
end


%% =========================================================================
%  Nullcline-Reconstruction SINDy: refit a plain, unconstrained thresholded
%  least squares regression after offsetting the data in phase space,
%  adapted from Prokop, Frolov & Gelens (2024). True K0, n are used for
%  the Hill column since this method has no mechanism for identifying an
%  embedded nonlinear parameter on its own.
%% =========================================================================
function [Xi_best, off_best, score_best] = run_NullclineSINDy_hes1( ...
        x, y, y_tau, dXdt, polyorder, K0, nHill)

    N = numel(x);
    cut = round(0.7*N);
    trIdx = 1:cut; valIdx = cut+1:N;

    objfun = @(off) nullcline_obj_hes1(off, x, y, y_tau, dXdt, polyorder, K0, nHill, trIdx, valIdx);
    off0 = [0, 0];
    opts = optimset('Display','off','MaxIter',80);
    off_best = fminsearch(objfun, off0, opts);
    [score_best, Xi_best] = nullcline_obj_hes1(off_best, x, y, y_tau, dXdt, polyorder, K0, nHill, trIdx, valIdx);
end

function [score, Xi] = nullcline_obj_hes1(off, x, y, y_tau, dXdt, polyorder, K0, nHill, trIdx, valIdx)
    xs = x + off(1); ys = y + off(2); yts = y_tau + off(2);

    Theta_full = [build_poly_library(xs, ys, polyorder), hill_func_local(yts, K0, nHill)];
    Theta_tr = Theta_full(trIdx,:);
    dY_tr = dXdt(trIdx,:);

    colscale = vecnorm(Theta_tr,2,1); colscale(colscale==0) = 1;
    ThetaN = Theta_tr ./ colscale;

    p = size(Theta_tr,2);
    Xi = zeros(p,2);
    % Equation 2 (protein) never sees the Hill column, same structural
    % exclusion as RI-SINDy and SR3 above.
    for eq = 1:2
        if eq == 1
            idx = 1:p;
        else
            idx = 1:p-1;
        end
        ThetaN_eq = ThetaN(:,idx);
        ys_ = dY_tr(:,eq);
        scaleY = norm(ys_,2); if scaleY==0, scaleY=1; end
        ysN = ys_/scaleY;
        xi_eq = ThetaN_eq \ ysN;
        for it = 1:10
            small = abs(xi_eq) < 0.1;
            big = ~small;
            if ~any(big)
                xi_eq = zeros(size(xi_eq)); break;
            end
            xi_eq = zeros(size(xi_eq));
            xi_eq(big) = ThetaN_eq(:,big) \ ysN;
        end
        xi_full = zeros(p,1);
        xi_full(idx) = xi_eq;
        Xi(:,eq) = (xi_full*scaleY) ./ colscale';
    end

    Theta_val = Theta_full(valIdx,:);
    pred = Theta_val * Xi;
    actual = dXdt(valIdx,:);
    ssres = sum((actual-pred).^2,'all');
    sstot = sum((actual-mean(actual,1)).^2,'all');
    R2 = 1 - ssres/max(sstot,1e-8);
    complexity = nnz(abs(Xi) > 1e-6);
    score = (1-R2) + 0.01*complexity;
end


%% =========================================================================
%  TRADITIONAL SINDy -- plain STLSQ, true K0/n given, no fairness
%  treatment at all: the Hill column is an ordinary candidate in BOTH
%  equations and is thresholded exactly like any polynomial term.
%% =========================================================================
function Xi = run_TraditionalSINDy_hes1(x, y, y_tau, dXdt, polyorder, K0, nHill)
    Theta = [build_poly_library(x, y, polyorder), hill_func_local(y_tau, K0, nHill)];
    p = size(Theta,2);
    nEq = size(dXdt,2);

    colscale = vecnorm(Theta,2,1); colscale(colscale==0) = 1;
    ThetaN = Theta ./ colscale;

    Xi = zeros(p, nEq);
    lambda = 0.1;   % STLSQ threshold, in normalized coefficient space
    for eq = 1:nEq
        ys_ = dXdt(:,eq);
        scaleY = norm(ys_,2); if scaleY==0, scaleY=1; end
        ysN = ys_/scaleY;
        xi = ThetaN \ ysN;
        for it = 1:15
            small = abs(xi) < lambda;
            big = ~small;
            if ~any(big)
                xi = zeros(size(xi)); break;
            end
            xi = zeros(size(xi));
            xi(big) = ThetaN(:,big) \ ysN;
        end
        Xi(:,eq) = (xi*scaleY) ./ colscale';
    end
end


%% =========================================================================
%  FORWARD INTEGRATION (delay-aware Euler step; Hes1's history dependence
%  rules out a plain ode45 call, so this stays local rather than using
%  paper/comparisons/utils/integrate_model.m). Returns a combined [nT x 2]
%  matrix rather than separate x_id/y_id, so it plugs directly into
%  mc_uq_propagate.m and print_error_block.m's generic [nT x n_dim] shape.
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