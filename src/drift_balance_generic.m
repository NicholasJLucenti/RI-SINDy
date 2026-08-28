function [pin_val, pin_var] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                      drain_cols, pinned_col, drain_basis, reg_basis, force_floor, force_multiplier)

% Every drift_balance_<system>.m file (Hes1, Goodwin, NF-kB, ...) does
% the same three things for each regulatory term: read the currently-
% estimated drain coefficients, average their physical magnitude over
% the data, and divide by the average magnitude of the regulatory
% term's own basis function to get the coefficient that balances them.
% This function IS that calculation. A system-specific
% drift_balance_<system>.m file's only job is to build drain_basis and
% reg_basis correctly and call this once per regulatory term (see the
% multi-term note below).


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