function [pin_val, pin_var] = drift_balance_nfkb(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                   x_data, z_data, hill_k0, hill_n)
% DRIFT_BALANCE_NFKB  NF-kB-specific drift-balance function: a single
% regulatory term, HillAct(x) -- an ACTIVATION Hill function (x^n /
% (k0^n + x^n)), the opposite shape from Hes1/Goodwin's repression Hill
% functions -- pinned only in the z (IkBa mRNA) equation. Neither the x
% nor y equations have a regulatory term at all.
%
%   [pin_val, pin_var] = DRIFT_BALANCE_NFKB(XiN_smooth, XiN_var, vi, ...
%                             col_scale, target_scale, x_data, z_data, hill_k0, hill_n)
%
% Library column layout this expects (see nfkb_risindy.m):
%   [1, S, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, x*y, Hill(x)]
%    1  2  3   4    5   6   7    8   9  10   11   12    13
%                                                         ^ pinned column
%
% Balances against the z-equation's OWN drain terms (z, z^2, z^3 --
% columns 9-11).
%
% Two things preserved from the original script that are NOT the same
% as Hes1/Goodwin's convention -- flagged here rather than silently
% changed:
%   1. A force multiplier (force_phys*1.05) -- this is a general
%      convergence aid, not an NF-kB-specific magic constant: when a
%      drift-balance fit is struggling to converge, nudging the balance
%      calculation with a multiplier like this can push it in the
%      direction it needs. Kept here at 1.05 since that's what worked
%      for this system; other systems can use a different value (or
%      none) via drift_balance_generic.m's force_multiplier argument.
%   2. The original computed avg_drain as abs(mean(...)) (average the
%      signed drain, then take the magnitude), NOT mean(abs(...)) (the
%      convention Hes1 and Goodwin now use, per an explicit correction
%      made earlier in this project). drift_balance_generic.m always
%      uses mean(abs(...)) -- so this file's numbers will differ
%      slightly from the literal original NF-kB script, in the same
%      direction and for the same reason as the Hes1 variance-scaling
%      fix: cancellation between positive and negative drain samples
%      shouldn't be allowed to deflate the estimate.
%
% See also: drift_balance_generic, risindy, nfkb_risindy.

    if vi ~= 3
        pin_val = 0;
        pin_var = 0;
        return;
    end

    drain_cols  = [9, 10, 11];             % z, z^2, z^3
    pinned_col  = 13;                       % Hill(x)
    drain_basis = [z_data, z_data.^2, z_data.^3];
    reg_basis   = (x_data.^hill_n) ./ (hill_k0^hill_n + x_data.^hill_n);   % activation form

    force_floor      = 1e-3;
    force_multiplier = 1.1;   % preserved from the original script

    [pin_val, pin_var] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                drain_cols, pinned_col, drain_basis, reg_basis, ...
                                                force_floor, force_multiplier);
end