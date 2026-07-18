%% =========================================================================
%  NF-kB Inflammatory Signaling Network
%  RI-SINDy (hardcoded, from a standalone nfkb_risindy.m run)  vs
%  SR3 (parametric-library, Champion et al. 2020)  vs
%  Nullcline-Reconstruction SINDy (Prokop, Frolov & Gelens 2024)  vs
%  Traditional SINDy (plain STLSQ, naive baseline)
%
%  RI-SINDy is NOT fit inside this script. Its coefficients come from a
%  standalone run of paper/nfkb_app/nfkb_risindy.m (the canonical
%  fitting script) and are hardcoded below, permuted into this script's
%  library column order, and only forward-integrated here for
%  comparison. This avoids running risindy.m twice per comparison run.
%
%  ----------------------------------------------------------------------
%  IMPLEMENTATION NOTE -- library column order mismatch:
%
%  nfkb_risindy.m (and drift_balance_nfkb.m, which it depends on) use a
%  DIFFERENT column order than this script's own library (nameLib,
%  used by SR3/Nullcline-SINDy/Traditional SINDy/forward integration/
%  error metrics/plotting below):
%     nfkb_risindy.m order:  [1, S, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, x*y, Hill(x)]
%     nameLib order:         [1, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, x*y, S(t), Hx]
%  See the detailed comment above the hardcoded Xi_ri block below for
%  the exact permutation applied.
%  ----------------------------------------------------------------------
%
%  Requires: Optimization Toolbox (lsqnonlin, fminsearch)
%
%  ----------------------------------------------------------------------
%  FAIRNESS NOTE (same structure as the Goodwin script, different from
%  the Hes1 script):
%
%  In Hes1, RI-SINDy structurally hardcodes the protein equation's Hill
%  coefficient to zero -- it is never even a regression candidate there.
%  Here, RI-SINDy does NOT do this. Every equation's candidate library
%  includes the regulatory term Hx, and RI-SINDy lets the sparsity step
%  correctly threshold it to zero in the equations it does not belong
%  to.
%
%  What RI-SINDy DOES force is that the z-equation's Hx term -- the one
%  assigned to it by the drift-balance condition -- is never subjected
%  to the sparsity threshold at all; its coefficient comes directly
%  from that condition rather than from a thresholded fit.
%
%  To match this fairly, SR3 and Nullcline-SINDy below give every
%  equation full access to the Hx column as an ordinary candidate, but
%  exempt the z-equation's Hx term from ever being zeroed by the
%  sparsity threshold, exactly mirroring what RI-SINDy already does.
%  Traditional SINDy applies NO such exemption to anything -- that is
%  what makes it the naive baseline.
%
%  Nullcline-SINDy is given the TRUE Hill hyperparameters (it has no
%  mechanism to discover them); SR3 jointly discovers them from a
%  generic, uninformed starting guess, since that is SR3's entire reason
%  for being tested.
%  ----------------------------------------------------------------------
% =========================================================================

close all; clc;
rng(11);
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, 'utils')));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

% Dock every figure into a single tabbed window instead of opening
% separate pop-up windows. To go back to normal floating windows, run
% set(0,'DefaultFigureWindowStyle','normal') in the command line.
set(0,'DefaultFigureWindowStyle','docked');

fprintf('=========================================================\n');
fprintf(' NF-kB Network: RI-SINDy vs SR3 vs Nullcline-SINDy vs Traditional SINDy\n');
fprintf('=========================================================\n');

sysDisplayName = 'NF-\kappaB Network';
[t, Xtrue, Xnoisy, p] = generate_nfkb_data();
fitIdx     = 1:numel(t);   % no train/extrapolate restriction established for NF-kB
ownRegIdx  = [0, 0, 13];   % eq3->Hx, eq1/eq2->none
libFun     = @(X,tt) lib_nfkb(X, tt, p.K0, p.n, p.S0, p.r);
nameLib    = {'1','x','x^2','x^3','y','y^2','y^3','z','z^2','z^3','x*y','S(t)','Hx'};
stateNames = {'NF-\kappaB (x)','I\kappaB (y)','IKK (z)'};

dt = t(2)-t(1);
fprintf('N = %d points total (fit on first %d), dt = %.3f\n', numel(t), numel(fitIdx), dt);

Xfit = Xnoisy(fitIdx,:);
dXdt_fit = derivative_sgolay(Xfit, dt);

% Library collinearity heatmap, on the same data every method fits on.
Theta_fit = libFun(Xfit, t(fitIdx));
plot_collinearity_heatmap(Theta_fit(:,2:end), nameLib(2:end), 'nfkb');

%% =========================================================================
%  RI-SINDy -- HARDCODED from a standalone nfkb_risindy.m run.
%
%  RI-SINDy is not fit inside this script (unlike SR3/Nullcline-SINDy/
%  Traditional SINDy below). This avoids running risindy.m twice --
%  nfkb_risindy.m is the canonical fitting script; this comparison
%  script's only job is to forward-integrate and compare against it.
%
%  IMPLEMENTATION NOTE -- library column order mismatch:
%  nfkb_risindy.m's own printed output (and drift_balance_nfkb.m, which
%  it depends on) use a DIFFERENT column order than this script's own
%  library (nameLib, used by SR3/Nullcline/Traditional/plotting/error
%  metrics):
%     nfkb_risindy.m order:  [1, S, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, x*y, Hill(x)]
%     nameLib order:         [1, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, x*y, S(t), Hx]
%  The coefficients and posterior variances below were manually permuted
%  from nfkb_risindy.m's printed order into nameLib order (S(t) moved
%  from position 2 to position 12) before being hardcoded here. Verify
%  against a fresh nfkb_risindy.m run before trusting these numbers if
%  that script's hyperparameters, data generation, or fitting options
%  ever change.
%
%  Source run:
%    NF-kB: 800 training samples, dt=0.0375
%    dx/dt = +0.9581*S -0.4241*x -1.0803*x*y
%    dy/dt = -0.2052*y +0.9731*z -0.8421*x*y
%    dz/dt = -0.6881*z +6.9522*Hill(x)
%    Fit metrics (eval t>3.0): x RMSE=0.0694 R^2=0.8541
%                              y RMSE=0.2837 R^2=0.8958
%                              z RMSE=0.2983 R^2=0.1046
%% =========================================================================
fprintf('\n--- Using hardcoded RI-SINDy model (from a standalone nfkb_risindy.m run) ---\n');

% rows in nameLib order: 1,x,x^2,x^3,y,y^2,y^3,z,z^2,z^3,x*y,S(t),Hx
Xi_ri = [   0,        0,       0      ; ...  % 1
           -0.4241,   0,       0      ; ...  % x
            0,        0,       0      ; ...  % x^2
            0,        0,       0      ; ...  % x^3
            0,       -0.2052,  0      ; ...  % y
            0,        0,       0      ; ...  % y^2
            0,        0,       0      ; ...  % y^3
            0,        0.9731, -0.6881 ; ...  % z
            0,        0,       0      ; ...  % z^2
            0,        0,       0      ; ...  % z^3
           -1.0803,  -0.8421,  0      ; ...  % x*y
            0.9581,   0,       0      ; ...  % S(t)
            0,        0,       6.9522 ];      % Hx

% Posterior variances, same nameLib row order as Xi_ri, permuted from
% nfkb_risindy.m's Xi_var output the same way as Xi_ri above.
Xi_ri_var = [ 0.0001, 0.0001, 0.0001 ; ...  % 1
              0.0062, 0.0004, 0.0005 ; ...  % x
              0.0001, 0.0003, 0.0002 ; ...  % x^2
              0.0000, 0.0009, 0.0001 ; ...  % x^3
              0.0030, 0.0049, 0.0001 ; ...  % y
              0.0018, 0.0000, 0.0000 ; ...  % y^2
              0.0001, 0.0000, 0.0000 ; ...  % y^3
              0.0001, 0.0255, 0.0012 ; ...  % z
              0.0000, 0.0013, 0.0000 ; ...  % z^2
              0.0000, 0.0000, 0.0000 ; ...  % z^3
              0.0022, 0.0215, 0.0002 ; ...  % x*y
              0.0010, 0.0001, 0.0001 ; ...  % S(t)
              0,      0,      0.0012 ];      % Hx

print_coeffs('RI-SINDy', Xi_ri, nameLib);
X0 = Xtrue(1,:);
rhs_ri = @(tt,X) (libFun(X', tt) * Xi_ri)';
Xhat_ri = integrate_model(rhs_ri, t, X0);

%% =========================================================================
%  SR3 -- parametric-library relax-and-split (Champion et al. 2020)
%% =========================================================================
fprintf('\n--- Fitting SR3 (jointly discovering Hill parameters) ---\n');
tic;
[Xi_sr3, K0_act, n_act] = run_SR3_nfkb(Xfit, t(fitIdx), dXdt_fit, p.S0, p.r, ownRegIdx);
fprintf('SR3 discovered: K0=%.3f, n=%.3f\n', K0_act, n_act);
fprintf('(True values: K0=%.2f, n=%d)\n', p.K0, p.n);
fprintf('SR3 finished in %.1f s\n', toc);
print_coeffs('SR3', Xi_sr3, nameLib);
rhs_sr3 = @(tt,X) nfkb_sr3_rhs(tt, X, Xi_sr3, K0_act, n_act, p.S0, p.r);
Xhat_sr3 = integrate_model(rhs_sr3, t, X0);

%% --- Nullcline-SINDy ----------------------------------------------------
fprintf('\n--- Fitting Nullcline-Reconstruction SINDy (true K0, n given) ---\n');
tic;
[Xi_null, off_best, score_best] = run_nullcline_sindy_3d(libFun, Xfit, t(fitIdx), dXdt_fit, ownRegIdx);
fprintf('Nullcline-SINDy finished in %.1f s\n', toc);
fprintf('Best offset: [%s]   score=%.4f\n', sprintf('%.3f ',off_best), score_best);
print_coeffs('Nullcline-SINDy', Xi_null, nameLib);
rhs_null = @(tt,X) (libFun((X'+off_best), tt) * Xi_null)';
Xhat_null = integrate_model(rhs_null, t, X0);

%% --- Traditional SINDy (naive baseline) ----------------------------------
fprintf('\n--- Fitting Traditional SINDy (plain STLSQ, true K0, n given) ---\n');
tic;
Xi_trad = run_traditional_sindy_3d(libFun, Xfit, t(fitIdx), dXdt_fit);
fprintf('Traditional SINDy finished in %.1f s\n', toc);
print_coeffs('Traditional SINDy', Xi_trad, nameLib);
rhs_trad = @(tt,X) (libFun(X', tt) * Xi_trad)';
Xhat_trad = integrate_model(rhs_trad, t, X0);

%% --- RI-SINDy UQ band via Monte Carlo over the coefficient posterior -----
fprintf('\n--- Propagating RI-SINDy UQ band (n=100 Monte Carlo draws) ---\n');
draw_fn = @(Xi_s) integrate_model(@(tt,X) (libFun(X', tt) * Xi_s)', t, X0);
[Xenv_lo, Xenv_hi] = mc_uq_propagate(draw_fn, Xi_ri, Xi_ri_var, 100);

%% --- Error metrics --------------------------------------------------------
fprintf('\n--- Error metrics vs. noise-free control trajectory ---\n');
print_error_block('RI-SINDy',          Xtrue, Xhat_ri,   stateNames);
print_error_block('SR3',               Xtrue, Xhat_sr3,  stateNames);
print_error_block('Nullcline-SINDy',   Xtrue, Xhat_null, stateNames);
print_error_block('Traditional SINDy', Xtrue, Xhat_trad, stateNames);

%% --- Method-by-state comparison grid (rows = states, cols = methods) -----
methodNames  = {'RI-SINDy','SR3','Nullcline-SINDy'};
methodTrajs  = {Xhat_ri, Xhat_sr3, Xhat_null};
methodColors = {[0 0.2 0.8], [0.75 0 0.55], [0.85 0.25 0]};

figure('Name', 'nfkb: method comparison grid', 'Color','w');
for d = 1:3
    rowVals = Xtrue(:,d);
    for m = 1:3
        tr = methodTrajs{m}(:,d);
        if all(isfinite(tr))
            rowVals = [rowVals; tr]; %#ok<AGROW>
        end
    end
    ylo = min(rowVals); yhi = max(rowVals);
    pad = 0.08*(yhi-ylo); if pad==0, pad = max(abs(yhi),1)*0.1; end
    rowYLim = [ylo-pad, yhi+pad];

    for m = 1:3
        subplot(3,3, (d-1)*3 + m); hold on;

        hUQ = [];
        if m == 1
            hUQ = fill([t; flipud(t)], [Xenv_hi(:,d); flipud(Xenv_lo(:,d))], ...
                methodColors{1}, 'FaceAlpha', 0.18, 'EdgeColor', 'none');
        end
        hTrue = plot(t, Xtrue(:,d), 'k-', 'LineWidth', 1);

        diverged = any(~isfinite(methodTrajs{m}(:,d)));
        hModel = [];
        if ~diverged
            hModel = plot(t, methodTrajs{m}(:,d), '--', 'Color', methodColors{m}, 'LineWidth', 1.1);
        end
        hSplit = xline(t(fitIdx(end)), 'k:', 'LineWidth', 0.6);

        xlim([t(1), t(end)]);
        ylim(rowYLim);
        if diverged
            text(mean(xlim), mean(rowYLim), 'model diverged', ...
                'HorizontalAlignment','center', 'FontSize', 8, ...
                'FontAngle','italic', 'Color',[0.45 0.45 0.45]);
        end
        if d == 1
            title(methodNames{m}, 'FontSize', 9, 'FontWeight','normal');
        end
        if m == 1
            ylabel(stateNames{d}, 'FontSize', 8);
        end
        if d == 3
            xlabel('time', 'FontSize', 8);
        end
        set(gca, 'FontSize', 7, 'Box','on');

        if d == 3 && m == 1
            legHandles = [hTrue, hModel, hUQ, hSplit];
            legLabels = {'Denoised trajectory','RI-SINDy','90% UQ band','train/extrap split'};
            if isempty(hModel)
                legHandles(2) = []; legLabels(2) = [];
            end
            legend(legHandles, legLabels, 'Location','best', 'FontSize',6, 'Box','on');
        end
    end
end
sgtitle(sysDisplayName, 'FontSize', 11, 'FontWeight','normal');

%% --- Phase portrait: data + RI-SINDy only ---------------------------------
figure('Name', 'nfkb: phase portrait', 'Color','w');
plot3(Xnoisy(:,1), Xnoisy(:,2), Xnoisy(:,3), '-', 'Color',[0.6 0.6 0.9], 'LineWidth',0.6);
hold on;
plot3(Xhat_ri(:,1), Xhat_ri(:,2), Xhat_ri(:,3), 'r--', 'LineWidth',1.5);
xlabel(stateNames{1}); ylabel(stateNames{2}); zlabel(stateNames{3});
legend({'Synthetic data','RI-SINDy model'}, 'Location','best');
title('Phase Portrait: NF-\kappaB');
grid on; view(135,25);


%% =========================================================================
%  DATA GENERATION
%% =========================================================================
function [t, Xtrue, Xnoisy, p] = generate_nfkb_data()
    p.kx = 1.0;  p.dx = 0.4;  p.dxy = 1.2;
    p.ky = 1.8;  p.dy = 0.3;
    p.alpha = 8.0;  p.K0 = 1.5;  p.n = 3;  p.dz = 0.8;
    p.S0 = 1.0;  p.r = 0.04;

    rhs = @(tt,X) nfkb_rhs(tt, X, p);
    tspan = (0:0.05:50)';
    X0 = [0.6; 0.6; 0.6];
    [t, Xtrue] = ode45(rhs, tspan, X0);

    noiseLevel = 0.10;
    sig = std(Xtrue,0,1);
    Xnoisy = Xtrue + noiseLevel .* sig .* randn(size(Xtrue));
end

function dX = nfkb_rhs(tt, X, p)
    x = X(1); y = X(2); z = X(3);
    S = p.S0 * exp(-p.r*tt);
    Hx = (x^p.n) / (p.K0^p.n + x^p.n);
    dx = p.kx*S - p.dxy*x*y - p.dx*x;
    dy = p.ky*z - p.dxy*x*y - p.dy*y;
    dz = p.alpha*Hx - p.dz*z;
    dX = [dx; dy; dz];
end

function dX = derivative_sgolay(X, dt)
    [m,n] = size(X);
    dX = zeros(m,n);
    for i = 1:n
        dX(:,i) = smooth_derivative(X(:,i), dt, 3, 11);
    end
end


%% =========================================================================
%  CANDIDATE LIBRARY (nameLib order -- used by SR3/Nullcline/Traditional/
%  forward integration/error metrics/plotting)
%% =========================================================================
function Theta = lib_nfkb(X, tt, K0, n, S0, r)
    x = X(:,1); y = X(:,2); z = X(:,3);
    tcol = tt(:);
    if numel(tcol) ~= size(X,1)
        tcol = repmat(tt, size(X,1), 1);
    end
    S  = S0 * exp(-r*tcol);
    Hx = (x.^n) ./ (K0^n + x.^n);
    Theta = [build_poly_library(x, y, z, 3), x.*y, S, Hx];
end


%% =========================================================================
%  SR3 -- NF-kB (one parametric regulatory column: Hx)
%% =========================================================================
function [Xi, K0_act, n_act] = run_SR3_nfkb(Xnoisy, t, dXdt, S0, r, ownRegIdx)

    nu = 0.05; lambda = 0.15; K_outer = 15;
    x = Xnoisy(:,1); y = Xnoisy(:,2); z = Xnoisy(:,3);
    S = S0*exp(-r*t(:));

    Theta_poly = [build_poly_library(x, y, z, 3), x.*y, S];
    p_poly = size(Theta_poly,2);
    nEq = 3;

    K0_act = 1.0; n_act = 2;   % generic, uninformed starting guess

    Xi = zeros(p_poly+1, nEq);
    W  = Xi;
    optsLSQ = optimoptions('lsqnonlin','Display','off','MaxIterations',60);

    for outer = 1:K_outer
        Hx_col = (x.^n_act) ./ (K0_act^n_act + x.^n_act);
        Theta = [Theta_poly, Hx_col];

        colscale = colscale_floor(vecnorm(Theta,2,1));
        ThetaN = Theta ./ colscale;
        targetScale = vecnorm(dXdt,2,1); targetScale(targetScale==0) = 1;
        dXdtN = dXdt ./ targetScale;

        WN = (W .* colscale') ./ targetScale;
        A = ThetaN'*ThetaN + eye(size(ThetaN,2))/nu;
        for eq = 1:nEq
            b = ThetaN'*dXdtN(:,eq) + WN(:,eq)/nu;
            xiN = A\b;
            Xi(:,eq) = (xiN .* targetScale(eq)) ./ colscale';
        end

        thresh = sqrt(2*lambda*nu) * (targetScale ./ mean(colscale));
        W = Xi;
        for eq = 1:nEq
            small = abs(W(:,eq)) < thresh(eq);
            if ownRegIdx(eq) > 0
                small(ownRegIdx(eq)) = false;
            end
            W(small, eq) = 0;
        end

        objfun = @(pp) sr3_nfkb_residual(pp, Theta_poly, x, Xi, dXdt);
        bestCost = inf; bestP = [K0_act, n_act];
        for trial = 1:3
            p0 = [0.5+9.5*rand(), 1+19*rand()];
            try
                [psol, cost] = lsqnonlin(objfun, p0, [0.5 1], [10 20], optsLSQ);
                if cost < bestCost
                    bestCost = cost; bestP = psol;
                end
            catch
                continue;
            end
        end
        K0_act = bestP(1); n_act = bestP(2);
    end
end

function res = sr3_nfkb_residual(p, Theta_poly, x, Xi, dXdt)
    K0_act = p(1); n_act = p(2);
    Hx_col = (x.^n_act) ./ (K0_act^n_act + x.^n_act);
    Theta = [Theta_poly, Hx_col];
    res = Theta*Xi - dXdt;
    res = res(:);

    % Prevents the optimizer from settling on a (K0, n) pair that drives
    % Hx to near-zero norm across NF-kB's narrow operating range, which
    % would otherwise blow up the recovered physical coefficient via
    % division by a near-zero column norm in the Xi update.
    penaltyWeight = 5;
    degPenalty = penaltyWeight / max(norm(Hx_col), 1e-6);
    res = [res; degPenalty];
end

function dX = nfkb_sr3_rhs(tt, X, Xi, K0_act, n_act, S0, r)
    x = X(1); y = X(2); z = X(3);
    S = S0*exp(-r*tt);
    Hx = (x^n_act) / (K0_act^n_act + x^n_act);
    row = [poly_library_row(x, y, z, 3), x*y, S, Hx];
    dX = (row*Xi)';
end