function [pin_val, pin_var] = drift_balance_hes1(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                   x_data, y_data, hill_k0, hill_n)
% DRIFT_BALANCE_HES1  Hes1-specific drift-balance function: balances the
% delayed Hill production term (mRNA equation only) against the mRNA
% drain terms (x, x^2). The protein equation has no Hill term at all --
% RI-SINDy hardcodes that coefficient to exactly zero rather than
% thresholding it out after the fact, which is the structural exclusion
% the SR3/Nullcline-SINDy comparison scripts explicitly mirror for
% fairness.
%
%   [pin_val, pin_var] = DRIFT_BALANCE_HES1(XiN_smooth, XiN_var, vi, ...
%                             col_scale, target_scale, x_data, y_data, hill_k0, hill_n)
%
% Library column layout this expects (see hes1_risindy.m):
%   [1, x, x^2, y, y^2, HillDelay(y_tau)]
%                                        ^ column 6, the pinned column
%
% x_data, y_data    training-window data (NOT delayed -- the Hill basis
%                    average here uses y_data directly, matching the
%                    original script's process, even though the actual
%                    library column is built from the delayed y_tau).
% hill_k0, hill_n    TRUE Hill hyperparameters (not fit).
%
% See also: drift_balance_generic, risindy, hes1_risindy.

    if vi == 2
        pin_val = 0;
        pin_var = 0;
        return;
    end

    drain_cols  = [2, 3];                          % x, x^2
    pinned_col  = 6;                                % HillDelay
    drain_basis = [x_data, x_data.^2];
    reg_basis   = 1 ./ (1 + (y_data./hill_k0).^hill_n);

    [pin_val, pin_var] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                drain_cols, pinned_col, drain_basis, reg_basis);
end