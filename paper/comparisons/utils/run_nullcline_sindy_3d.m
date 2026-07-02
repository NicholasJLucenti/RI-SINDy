function [Xi_best, off_best, score_best] = run_nullcline_sindy_3d(libFun, Xnoisy, t, dXdt, ownRegIdx)
% RUN_NULLCLINE_SINDY_3D  Nullcline-Reconstruction SINDy (Prokop, Frolov
% & Gelens 2024, adapted), generalized to any 3-state system. Used
% as-is by both Goodwin and NF-kB -- nothing here is system-specific.
%
%   [Xi_best, off_best, score_best] = RUN_NULLCLINE_SINDY_3D(libFun, Xnoisy, t, dXdt, ownRegIdx)
%
% Searches for a phase-space offset that best aligns the (noisy) data
% with its true nullclines, then refits a plain thresholded least
% squares regression at that offset. Has no mechanism for identifying
% an embedded nonlinear regulatory parameter itself -- libFun must
% already have the TRUE regulatory hyperparameters baked in.
%
% INPUTS
%   libFun     function handle: Theta = libFun(X, t), X is [N x 3],
%              Theta is [N x n_lib]. Column 1 of Theta MUST be the
%              constant term.
%   Xnoisy     [N x 3]  noisy state data.
%   t          [N x 1]  time points matching Xnoisy.
%   dXdt       [N x 3]  numerical derivative target.
%   ownRegIdx  [1 x 3]  for each equation, the column index of the ONE
%              regulatory term structurally assigned to it (0 if none).
%              That column is exempt from the sparsity threshold in its
%              own equation; every other column (including another
%              equation's regulatory term, if present as an ordinary
%              candidate here) is thresholded normally.
%
% OUTPUTS
%   Xi_best     [n_lib x 3]  coefficients at the best-found offset.
%   off_best    [1 x 3]  the phase-space offset found.
%   score_best  objective value at off_best (lower is better; combines
%               held-out R^2 and model complexity).
%
% See also: run_traditional_sindy_3d, plot_collinearity_heatmap.

    N = size(Xnoisy,1);
    cut = round(0.7*N);
    trIdx = 1:cut; valIdx = cut+1:N;

    objfun = @(off) nullcline_obj_3d(off, libFun, Xnoisy, t, dXdt, trIdx, valIdx, ownRegIdx);
    off0 = zeros(1,3);
    opts = optimset('Display','off','MaxIter',80);
    off_best = fminsearch(objfun, off0, opts);
    [score_best, Xi_best] = nullcline_obj_3d(off_best, libFun, Xnoisy, t, dXdt, trIdx, valIdx, ownRegIdx);
end

function [score, Xi] = nullcline_obj_3d(off, libFun, X, t, dXdt, trIdx, valIdx, ownRegIdx)
    Xshift = X + off;
    Theta_tr = libFun(Xshift(trIdx,:), t(trIdx));
    dY_tr = dXdt(trIdx,:);
    nEq = size(dY_tr,2);
    p = size(Theta_tr,2);

    colscale = vecnorm(Theta_tr,2,1); colscale(colscale==0) = 1;
    ThetaN = Theta_tr ./ colscale;

    Xi = zeros(p,nEq);
    for eq = 1:nEq
        ys_ = dY_tr(:,eq);
        scaleY = norm(ys_,2); if scaleY==0, scaleY=1; end
        ysN = ys_/scaleY;
        xi = ThetaN \ ysN;
        for it = 1:10
            small = abs(xi) < 0.1;
            if ownRegIdx(eq) > 0
                small(ownRegIdx(eq)) = false;
            end
            big = ~small;
            if ~any(big)
                xi = zeros(size(xi)); break;
            end
            xi = zeros(size(xi));
            xi(big) = ThetaN(:,big) \ ysN;
        end
        Xi(:,eq) = (xi*scaleY) ./ colscale';
    end

    Theta_val = libFun(Xshift(valIdx,:), t(valIdx));
    pred = Theta_val*Xi;
    actual = dXdt(valIdx,:);
    ssres = sum((actual-pred).^2,'all');
    sstot = sum((actual-mean(actual,1)).^2,'all');
    R2 = 1 - ssres/max(sstot,1e-8);
    complexity = nnz(abs(Xi)>1e-6);
    score = (1-R2) + 0.01*complexity;
end