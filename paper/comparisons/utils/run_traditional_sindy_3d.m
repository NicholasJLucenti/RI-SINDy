function Xi = run_traditional_sindy_3d(libFun, X, t, dXdt)
% RUN_TRADITIONAL_SINDY_3D  Naive baseline: plain STLSQ, true
% hyperparameters already baked into libFun, NO fairness treatment at
% all -- every column, including any regulatory term, competes equally
% and is thresholded the same way as any polynomial term. Used as-is by
% both Goodwin and NF-kB.
%
%   Xi = RUN_TRADITIONAL_SINDY_3D(libFun, X, t, dXdt)
%
% INPUTS
%   libFun   function handle: Theta = libFun(X, t) (same contract as
%            run_nullcline_sindy_3d.m).
%   X, t     [N x 3], [N x 1]  state data and matching time points.
%   dXdt     [N x 3]  numerical derivative target.
%
% OUTPUT
%   Xi   [n_lib x 3]  identified coefficients.
%
% See also: run_nullcline_sindy_3d.

    Theta = libFun(X, t);
    nEq = size(dXdt,2);
    p = size(Theta,2);

    colscale = vecnorm(Theta,2,1); colscale(colscale==0) = 1;
    ThetaN = Theta ./ colscale;

    Xi = zeros(p, nEq);
    lambda = 0.1;   % STLSQ threshold, in normalized coefficient space
    for eq = 1:nEq
        ys = dXdt(:,eq);
        scaleY = norm(ys,2); if scaleY==0, scaleY=1; end
        ysN = ys/scaleY;
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