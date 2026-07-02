function [pin_val, pin_var] = drift_balance_goodwin(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                                      x_data, y_data, z_data, hill_k0, hill_n, hill_km)
% DRIFT_BALANCE_GOODWIN  Goodwin-specific drift-balance function: two
% independent regulatory terms, each assigned to a different equation.
%   - HillRep(z) (column 11) balances the mRNA equation's drain terms
%     (x, x^2, x^3).
%   - HillDeg(y) (column 12) balances the protein equation's production
%     terms -- the SAME column indices [2,3,4] (x, x^2, x^3), but read
%     from the protein equation's own coefficients, since vi=2 there.
% The repressor equation (vi=3) has no regulatory term at all.
%
%   [pin_val, pin_var] = DRIFT_BALANCE_GOODWIN(XiN_smooth, XiN_var, vi, ...
%                             col_scale, target_scale, x_data, y_data, z_data, ...
%                             hill_k0, hill_n, hill_km)
%
% Library column layout this expects (see goodwin_risindy.m):
%   [1, x, x^2, x^3, y, y^2, y^3, z, z^2, z^3, HillRep(z), HillDeg(y)]
%                                                11           12
%
% Returns a 2-element pin_val/pin_var (one entry per pinned column, in
% the order [HillRep(z); HillDeg(y)]), with 0 for whichever regulatory
% term is inactive at this vi.
%
% See also: drift_balance_generic, risindy, goodwin_risindy.

    pin_val = zeros(2,1);
    pin_var = zeros(2,1);

    switch vi
        case 1   % mRNA equation: HillRep(z) balances the drain (x, x^2, x^3)
            drain_cols  = [2, 3, 4];
            drain_basis = [x_data, x_data.^2, x_data.^3];
            reg_basis   = (hill_k0^hill_n) ./ (hill_k0^hill_n + z_data.^hill_n);
            [pv, pr] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                              drain_cols, 11, drain_basis, reg_basis);
            pin_val(1) = pv;
            pin_var(1) = pr;

        case 2   % protein equation: HillDeg(y) balances the production (x, x^2, x^3)
            drain_cols  = [2, 3, 4];
            drain_basis = [x_data, x_data.^2, x_data.^3];
            reg_basis   = y_data ./ (hill_km + y_data);
            [pv, pr] = drift_balance_generic(XiN_smooth, XiN_var, vi, col_scale, target_scale, ...
                                              drain_cols, 12, drain_basis, reg_basis);
            % HillDeg is a LOSS term here (degradation), so its
            % coefficient must be negative even though the balance
            % magnitude computed above is always non-negative.
            pin_val(2) = -pv;
            pin_var(2) = pr;

        case 3   % repressor equation: no regulatory term at all
            % pin_val, pin_var stay zero -- see initialization above.
    end
end