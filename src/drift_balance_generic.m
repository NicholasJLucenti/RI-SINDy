function [pin_val, pin_var] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                      drain_cols, pinned_col, drain_basis, reg_basis, force_floor, force_multiplier)
% DRIFT_BALANCE_GENERIC  Reusable core of the drift-balance calculation.
%
%   [pin_val, pin_var] = DRIFT_BALANCE_GENERIC(XiN_smooth, XiN_var, vi, ...
%                             col_scale, target_scale, drain_cols, pinned_col, ...
%                             drain_basis, reg_basis)
%   [...] = DRIFT_BALANCE_GENERIC(..., force_floor)
%   [...] = DRIFT_BALANCE_GENERIC(..., force_floor, force_multiplier)
%
% Every drift_balance_<system>.m file (Hes1, Goodwin, NF-kB, ...) does
% the same three things for each regulatory term: read the currently-
% estimated drain coefficients, average their physical magnitude over
% the data, and divide by the average magnitude of the regulatory
% term's own basis function to get the coefficient that balances them.
% This function IS that calculation. A system-specific
% drift_balance_<system>.m file's only job is to build drain_basis and
% reg_basis correctly and call this once per regulatory term (see the
% multi-term note below).
%
% INPUTS
%   XiN_smooth   [n_lib x n_var]  smoothed normalized coefficients, as
%                passed into your drift_fn by risindy.m.
%   XiN_var      [n_lib x n_var]  raw (unsmoothed) normalized-space
%                coefficient variances, as passed into your drift_fn.
%   vi           which equation is currently being fit.
%   col_scale    [1 x n_lib]  from risindy.m's drift_fn call.
%   target_scale [1 x n_var]  from risindy.m's drift_fn call.
%   drain_cols   column indices (into Theta) of the polynomial "drain"
%                terms this regulatory term is being balanced against
%                -- e.g. [2, 3] for [x, x^2].
%   pinned_col   the SINGLE column index of the regulatory term being
%                solved for right now (a scalar -- see multi-term note).
%   drain_basis  [N x numel(drain_cols)] the RAW (unnormalized) basis
%                values for each drain column, same column order as
%                drain_cols -- e.g. [x_data, x_data.^2].
%   reg_basis    [N x 1] the RAW (unnormalized) regulatory basis values
%                over the data -- e.g. hill_func(y_tau).
%   force_floor  (optional) floor on the average regulatory basis
%                magnitude, to avoid dividing by ~0. Default 1e-3.
%   force_multiplier  (optional) scalar multiplier applied to the drain
%                magnitude before dividing by the regulatory basis.
%                This is a convergence aid: if a drift-balance fit is
%                struggling to settle (the pinned coefficient consistently
%                lands short of where it needs to be), nudging this away
%                from 1 can push the balance calculation in the right
%                direction -- e.g. NF-kB's script uses 1.05. Not a
%                per-system constant to leave untouched; treat it as a
%                knob to reach for on a new system before assuming
%                something else is wrong. Default 1 (no adjustment).
%
% OUTPUTS
%   pin_val, pin_var   scalars, already in NORMALIZED (XiN) space --
%                      pass straight through as one entry of the vectors
%                      your drift_fn returns to risindy.m.
%
% MULTI-TERM SYSTEMS (e.g. Goodwin, which has HillRep(z) pinned in the
% x-equation and HillDeg(y) pinned in the y-equation): call this once
% per regulatory term, each with its own drain_cols/pinned_col/
% drain_basis/reg_basis, then assemble the outputs into the pin_val/
% pin_var vectors risindy.m expects (using 0 for the entries where a
% given equation has no active regulatory term). See
% drift_balance_template.m for the full vector-assembly contract.
%
% NOTE ON SIGN CONVENTION: this uses mean(abs(...)), i.e. it takes the
% per-sample magnitude of the drain contribution first, then averages
% those magnitudes -- NOT abs(mean(...)). We're measuring a typical
% magnitude to balance against, so cancellation between positive and
% negative samples (which abs(mean(...)) would allow) should not be
% able to deflate the estimate.
%
% See also: risindy, drift_balance_template.

    if nargin < 10, force_floor = 1e-3; end
    if nargin < 11, force_multiplier = 1; end

    n_drain = numel(drain_cols);
    cx      = zeros(n_drain, 1);
    for j = 1:n_drain
        cx(j) = (XiN_smooth(drain_cols(j), vi) * target_scale(vi)) / col_scale(drain_cols(j));
    end

    avg_drain   = mean(abs(drain_basis * cx));
    h_basis_avg = mean(reg_basis);
    force_phys  = (force_multiplier * avg_drain) / max(h_basis_avg, force_floor);

    pin_val = (force_phys * col_scale(pinned_col)) / target_scale(vi);

    var_drain   = XiN_var(drain_cols, vi);
    scaling_vec = col_scale(pinned_col) ./ col_scale(drain_cols)';
    pin_var     = sum(var_drain .* scaling_vec(:).^2);
end