function [Xi, incl_prob] = run_ensemble_sindy_3d(libFun, X, t, dXdt, n_boot, incl_threshold, stlsq_lambda)
% RUN_ENSEMBLE_SINDY_3D  Bootstrap-ensemble SINDy (E-SINDy, "bagging"
% variant per Fasel, Kutz, Brunton & Proctor 2022), generalized to any
% 3-state system. Used as-is by any noise-sensitivity or robustness
% comparison -- nothing here is system-specific.
%
%   [Xi, incl_prob] = RUN_ENSEMBLE_SINDY_3D(libFun, X, t, dXdt)
%   [...] = RUN_ENSEMBLE_SINDY_3D(libFun, X, t, dXdt, n_boot, incl_threshold, stlsq_lambda)
%
% Fits plain STLSQ (same algorithm as run_traditional_sindy_3d.m)
% independently on n_boot bootstrap resamples of the training data (rows
% resampled WITH replacement), then keeps a library term only if it
% survives thresholding in at least incl_threshold fraction of the
% bootstrap fits -- its "inclusion probability." The kept coefficient is
% the MEDIAN across the bootstraps where that term survived, which is
% more robust to any single noise-driven outlier fit than one STLSQ run
% -- that robustness is the entire reason to test this method at all.
%
% INPUTS
%   libFun          function handle: Theta = libFun(X, t) (same contract
%                   as run_nullcline_sindy_3d.m / run_traditional_sindy_3d.m).
%   X, t            [N x 3], [N x 1]  state data and matching time points.
%   dXdt            [N x 3]  numerical derivative target.
%   n_boot          (optional) number of bootstrap resamples. Default 100.
%   incl_threshold  (optional) minimum fraction of bootstraps a term must
%                   survive in to be kept. Default 0.5 (majority vote).
%   stlsq_lambda    (optional) STLSQ threshold applied within each
%                   bootstrap fit. Default 0.1.
%
% OUTPUTS
%   Xi         [n_lib x 3]  final coefficients (0 wherever inclusion
%              probability fell below incl_threshold).
%   incl_prob  [n_lib x 3]  fraction of bootstraps each term survived in
%              -- useful for inspecting how close a term was to the
%              threshold, not just whether it made the final cut.
%
% NOTE: this is a bootstrap-over-DATA-ROWS ensemble ("bragging"). The
% original E-SINDy paper also describes a library-bootstrap variant;
% this file implements only the data-row version, which is the simpler
% and more commonly cited one for noise-robustness comparisons.
%
% See also: run_traditional_sindy_3d, count_recovered_terms.

    if nargin < 5, n_boot = 100; end
    if nargin < 6, incl_threshold = 0.5; end
    if nargin < 7, stlsq_lambda = 0.1; end

    N     = size(X, 1);
    n_lib = size(libFun(X, t), 2);
    nEq   = size(dXdt, 2);

    coeffs_boot   = zeros(n_lib, nEq, n_boot);
    survived_boot = false(n_lib, nEq, n_boot);

    for b = 1:n_boot
        idx = randi(N, N, 1);   % resample rows with replacement
        Xb    = X(idx, :);
        tb    = t(idx);
        dXdtb = dXdt(idx, :);

        Theta_b  = libFun(Xb, tb);
        colscale = vecnorm(Theta_b, 2, 1); colscale(colscale==0) = 1;
        ThetaN_b = Theta_b ./ colscale;

        for eq = 1:nEq
            ys = dXdtb(:, eq);
            scaleY = norm(ys, 2); if scaleY == 0, scaleY = 1; end
            ysN = ys / scaleY;
            xi = ThetaN_b \ ysN;
            for it = 1:15
                small = abs(xi) < stlsq_lambda;
                big = ~small;
                if ~any(big)
                    xi = zeros(size(xi)); break;
                end
                xi = zeros(size(xi));
                xi(big) = ThetaN_b(:, big) \ ysN;
            end
            xi_phys = (xi * scaleY) ./ colscale';
            coeffs_boot(:, eq, b)   = xi_phys;
            survived_boot(:, eq, b) = abs(xi_phys) > 1e-8;
        end
    end

    incl_prob = mean(survived_boot, 3);
    Xi = zeros(n_lib, nEq);
    for eq = 1:nEq
        for j = 1:n_lib
            if incl_prob(j, eq) >= incl_threshold
                surv_vals = squeeze(coeffs_boot(j, eq, :));
                surv_vals = surv_vals(squeeze(survived_boot(j, eq, :)));
                if ~isempty(surv_vals)
                    Xi(j, eq) = median(surv_vals);
                end
            end
        end
    end
end