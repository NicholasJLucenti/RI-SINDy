%% =========================================================================
%  Goodwin Oscillator
%  SR3 (parametric-library, Champion et al. 2020)  vs
%  Nullcline-Reconstruction SINDy (Prokop, Frolov & Gelens 2024)  vs
%  Traditional SINDy (plain STLSQ, naive baseline)
%
%  RI-SINDy is NOT fit here -- its coefficients are hardcoded from the
%  already-published table for forward integration and comparison only.
%
%  Requires: Optimization Toolbox (lsqnonlin, fminsearch)
%
%  ----------------------------------------------------------------------
%  FAIRNESS NOTE (different from the Hes1 script):
%
%  In Hes1, RI-SINDy structurally hardcodes the protein equation's Hill
%  coefficient to zero -- it is never even a regression candidate there.
%  Here, RI-SINDy does NOT do this. Every equation's candidate library
%  includes BOTH regulatory terms (Hrep(z) and Hdeg(y)), and RI-SINDy
%  lets the sparsity step correctly threshold the "foreign" regulatory
%  term to zero on its own.
%
%  What RI-SINDy DOES force is that each equation's OWN regulatory term
%  -- the one assigned to it by the drift-balance condition -- is never
%  subjected to the sparsity threshold at all; its coefficient comes
%  directly from that condition rather than from a thresholded fit.
%
%  To match this fairly, SR3 and Nullcline-SINDy below give every
%  equation full access to both regulatory columns as ordinary
%  candidates, but exempt each equation's own assigned regulatory term
%  from ever being zeroed by the sparsity threshold, exactly mirroring
%  what RI-SINDy already does. Traditional SINDy applies NO such
%  exemption to anything -- that is what makes it the naive baseline.
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
fprintf(' Goodwin Oscillator: SR3 vs Nullcline-SINDy vs Traditional SINDy\n');
fprintf('=========================================================\n');

sysDisplayName = 'Goodwin Oscillator';
[t, Xtrue, Xnoisy, p, N_train] = generate_goodwin_data();
fitIdx     = 1:N_train;   % RI-SINDy itself was only ever fit on this window
ownRegIdx  = [11, 12, 0]; % eq1->Hrep(z), eq2->Hdeg(y), eq3->none
libFun     = @(X,tt) lib_goodwin(X, tt, p.K0, p.n, p.Km);
nameLib    = {'1','x','x^2','x^3','y','y^2','y^3','z','z^2','z^3','Hrep(z)','Hdeg(y)'};
stateNames = {'mRNA (x)','Protein (y)','Repressor (z)'};

% RI-SINDy coefficients from the published table, column order matching
% nameLib above. Rows = library terms, cols = mRNA/Protein/Repressor.
Xi_ri = [   0,        0,       0      ; ...  % 1
           -0.7518,   2.0230,  0      ; ...  % x
            0,        0,       0      ; ...  % x^2
            0,        0,       0      ; ...  % x^3
            0,        0,       1.4857 ; ...  % y
            0,        0,       0      ; ...  % y^2
            0,        0,       0      ; ...  % y^3
            0,        0,      -0.7906 ; ...  % z
            0,        0,       0      ; ...  % z^2
            0,        0,       0      ; ...  % z^3
            9.3623,   0,       0      ; ...  % Hrep(z)
            0,       -5.2834,  0      ];      % Hdeg(y)

% Posterior standard deviations (physical units), same 12x3 layout as
% Xi_ri, squared into variance for the Monte Carlo UQ propagation.
Xi_ri_std = 2*[ 0.0022, 0.0033, 0.0041 ; ...  % 1
              0.0102, 0.0151, 0.0020 ; ...  % x
              0.0003, 0.0005, 0.0006 ; ...  % x^2
              0.0001, 0.0001, 0.0002 ; ...  % x^3
              0.0005, 0.0008, 0.0194 ; ...  % y
              0.0001, 0.0001, 0.0002 ; ...  % y^2
              0.0000, 0.0000, 0.0000 ; ...  % y^3
              0.0003, 0.0005, 0.0115 ; ...  % z
              0.0000, 0.0001, 0.0001 ; ...  % z^2
              0.0000, 0.0000, 0.0000 ; ...  % z^3
              0.0102, 0,      0      ; ...  % Hrep(z)
              0,      0.0151, 0      ];      % Hdeg(y)
Xi_ri_var = Xi_ri_std.^2;

dt = t(2)-t(1);
fprintf('N = %d points total (fit on first %d), dt = %.3f\n', numel(t), numel(fitIdx), dt);

%% --- SR3 --------------------------------------------------------------
fprintf('\n--- Fitting SR3 (jointly discovering Hill parameters) ---\n');
tic;
Xfit = Xnoisy(fitIdx,:);
dXdt_fit = derivative_sgolay(Xfit, dt);

% Library collinearity heatmap, on the same data every method fits on.
Theta_fit = libFun(Xfit, t(fitIdx));
plot_collinearity_heatmap(Theta_fit(:,2:end), nameLib(2:end), 'goodwin');

[Xi_sr3, K0_rep, n_rep, Km_deg] = run_SR3_goodwin(Xfit, dXdt_fit, ownRegIdx);
fprintf('SR3 discovered: K0_rep=%.3f, n_rep=%.3f, Km_deg=%.3f\n', K0_rep, n_rep, Km_deg);
fprintf('(True values: K0=%.2f, n=%d, Km=%.2f)\n', p.K0, p.n, p.Km);
fprintf('SR3 finished in %.1f s\n', toc);
print_coeffs('SR3', Xi_sr3, nameLib);
X0 = Xtrue(1,:);
rhs_sr3 = @(tt,X) goodwin_sr3_rhs(X, Xi_sr3, K0_rep, n_rep, Km_deg);
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

%% --- Propagate the already-identified RI-SINDy model ----------------------
fprintf('\n--- Propagating the already-identified RI-SINDy model ---\n');
print_coeffs('RI-SINDy', Xi_ri, nameLib);
rhs_ri = @(tt,X) (libFun(X', tt) * Xi_ri)';
Xhat_ri = integrate_model(rhs_ri, t, X0);

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
% Each panel holds exactly the true dynamics and one method's
% trajectory, nothing else. Traditional SINDy and the noisy data
% scatter are deliberately excluded here (both stay fully represented
% in the tables and error blocks above) -- matching the original design
% intent of only visually contrasting RI-SINDy against the two targeted
% prior-art comparisons. RI-SINDy's column additionally shows its 90%
% UQ band, since it is the only method here with a posterior to draw
% one from. If a method's trajectory diverged, the panel says so
% explicitly rather than silently rendering empty.
methodNames  = {'RI-SINDy','SR3','Nullcline-SINDy'};
methodTrajs  = {Xhat_ri, Xhat_sr3, Xhat_null};
methodColors = {[0 0.2 0.8], [0.75 0 0.55], [0.85 0.25 0]};

figure('Name', 'goodwin: method comparison grid', 'Color','w');
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
figure('Name', 'goodwin: phase portrait', 'Color','w');
plot3(Xnoisy(:,1), Xnoisy(:,2), Xnoisy(:,3), '-', 'Color',[0.6 0.6 0.9], 'LineWidth',0.6);
hold on;
plot3(Xhat_ri(:,1), Xhat_ri(:,2), Xhat_ri(:,3), 'r--', 'LineWidth',1.5);
xlabel(stateNames{1}); ylabel(stateNames{2}); zlabel(stateNames{3});
legend({'Synthetic data','RI-SINDy model'}, 'Location','best');
title('Phase Portrait: Goodwin');
grid on; view(135,25);


%% =========================================================================
%  DATA GENERATION
%% =========================================================================
function [t, Xtrue, Xnoisy, p, N_train] = generate_goodwin_data()
    p.alpha = 8.0;  p.dx = 0.6;  p.K0 = 2.0;  p.n = 3;
    p.ky    = 2.0;
    p.beta  = 5.0;  p.Km = 1.0;
    p.kz    = 1.5;  p.dz = 0.8;

    dt = 0.05;
    N_train = 300;
    n_total = N_train + 200;

    rhs = @(tt,X) goodwin_rhs(X, p);
    tspan = (0 : dt : n_total*dt)';
    X0 = [1.5; 0.5; 1.0];
    [t, Xtrue] = ode45(rhs, tspan, X0, odeset('RelTol',1e-9,'AbsTol',1e-11));

    noiseLevel = 0.05;
    sig = std(Xtrue,0,1);
    Xnoisy = Xtrue + noiseLevel .* sig .* randn(size(Xtrue));
end

function dX = goodwin_rhs(X, p)
    x = X(1); y = X(2); z = X(3);
    Hrep = (p.K0^p.n) / (p.K0^p.n + z^p.n);
    Hdeg = y / (p.Km + y);
    dx = p.alpha*Hrep - p.dx*x;
    dy = p.ky*x - p.beta*Hdeg;
    dz = p.kz*y - p.dz*z;
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
%  CANDIDATE LIBRARY
%% =========================================================================
function Theta = lib_goodwin(X, tt, K0, n, Km)
    x = X(:,1); y = X(:,2); z = X(:,3);
    Hrep = (K0^n) ./ (K0^n + z.^n);
    Hdeg = y ./ (Km + y);
    Theta = [build_poly_library(x, y, z, 3), Hrep, Hdeg];
end


%% =========================================================================
%  SR3 -- Goodwin (two parametric regulatory columns: Hrep, Hdeg)
%% =========================================================================
function [Xi, K0_rep, n_rep, Km_deg] = run_SR3_goodwin(Xnoisy, dXdt, ownRegIdx)

    nu = 0.05; lambda = 0.15; K_outer = 15;
    x = Xnoisy(:,1); y = Xnoisy(:,2); z = Xnoisy(:,3);

    Theta_poly = build_poly_library(x, y, z, 3);
    p_poly = size(Theta_poly,2);
    nEq = 3;

    K0_rep = 1.0; n_rep = 2; Km_deg = 2.0;   % generic, uninformed starting guess

    Xi = zeros(p_poly+2, nEq);
    W  = Xi;
    optsLSQ = optimoptions('lsqnonlin','Display','off','MaxIterations',60);

    for outer = 1:K_outer
        Hrep_col = (K0_rep^n_rep) ./ (K0_rep^n_rep + z.^n_rep);
        Hdeg_col = y ./ (Km_deg + y);
        Theta = [Theta_poly, Hrep_col, Hdeg_col];

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

        % Each equation's OWN regulatory term is exempt from
        % thresholding, matching RI-SINDy; the "foreign" regulatory
        % term is a normal candidate and can be thresholded to zero.
        thresh = sqrt(2*lambda*nu) * (targetScale ./ mean(colscale));
        W = Xi;
        for eq = 1:nEq
            small = abs(W(:,eq)) < thresh(eq);
            if ownRegIdx(eq) > 0
                small(ownRegIdx(eq)) = false;
            end
            W(small, eq) = 0;
        end

        objfun = @(pp) sr3_goodwin_residual(pp, Theta_poly, z, y, Xi, dXdt);
        bestCost = inf; bestP = [K0_rep, n_rep, Km_deg];
        for trial = 1:3
            p0 = [0.5+9.5*rand(), 1+19*rand(), 0.1+9.9*rand()];
            try
                [psol, cost] = lsqnonlin(objfun, p0, [0.5 1 0.1], [10 20 10], optsLSQ);
                if cost < bestCost
                    bestCost = cost; bestP = psol;
                end
            catch
                continue;
            end
        end
        K0_rep = bestP(1); n_rep = bestP(2); Km_deg = bestP(3);
    end
end

function res = sr3_goodwin_residual(p, Theta_poly, z, y, Xi, dXdt)
    K0_rep = p(1); n_rep = p(2); Km_deg = p(3);
    Hrep_col = (K0_rep^n_rep) ./ (K0_rep^n_rep + z.^n_rep);
    Hdeg_col = y ./ (Km_deg + y);
    Theta = [Theta_poly, Hrep_col, Hdeg_col];
    res = Theta*Xi - dXdt;
    res = res(:);

    % Prevents the optimizer from settling on a (K0, n) or Km pair that
    % drives a regulatory column to near-zero norm, which would
    % otherwise blow up the recovered physical coefficient via division
    % by a near-zero norm in the Xi update.
    penaltyWeight = 5;
    repPenalty = penaltyWeight / max(norm(Hrep_col), 1e-6);
    degPenalty = penaltyWeight / max(norm(Hdeg_col), 1e-6);
    res = [res; repPenalty; degPenalty];
end

function dX = goodwin_sr3_rhs(X, Xi, K0_rep, n_rep, Km_deg)
    x = X(1); y = X(2); z = X(3);
    Hrep = (K0_rep^n_rep) / (K0_rep^n_rep + z^n_rep);
    Hdeg = y / (Km_deg + y);
    row = [poly_library_row(x, y, z, 3), Hrep, Hdeg];
    dX = (row*Xi)';
end