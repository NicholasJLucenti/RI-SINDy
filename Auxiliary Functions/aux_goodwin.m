function [pin_val, pin_var] = aux_goodwin(XiN, XiN_var, vi, col_scale, target_scale, ...
                                           x_train, y_train, z_train, hill_k0, hill_n, hill_km)
% Pinned auxiliary function for the Goodwin oscillator.
% pinned_cols = [hill_rep_col, hill_deg_col] (last two columns of Theta)
% Returns pin_val and pin_var as [2x1] vectors matching pinned_cols.
%
%   vi=1 (dx/dt): HillRep(z) active, HillDeg zero
%   vi=2 (dy/dt): HillRep zero,      HillDeg(y) active
%   vi=3 (dz/dt): both zero

    n_poly       = size(XiN, 1) - 2;   % Hill terms are last two rows
    hill_rep_col = n_poly + 1;
    hill_deg_col = n_poly + 2;

    switch vi

        case 1
            % Balance HillRep(z) production against x-drain terms
            cx  = (XiN(2, 1) * target_scale(1)) / col_scale(2);
            cx2 = (XiN(3, 1) * target_scale(1)) / col_scale(3);
            cx3 = (XiN(4, 1) * target_scale(1)) / col_scale(4);

            avg_drain   = mean(abs(cx*x_train + cx2*x_train.^2 + cx3*x_train.^3));
            h_basis_avg = mean(hill_k0^hill_n ./ (hill_k0^hill_n + z_train.^hill_n));
            force_phys  = avg_drain / max(h_basis_avg, 1e-3);

            pv = (force_phys * col_scale(hill_rep_col)) / target_scale(1);
            s  = (target_scale(1) ./ col_scale(2:4)') .* (col_scale(hill_rep_col) / target_scale(1));
            pvar = sum(XiN_var(2:4, 1) .* s.^2);

            pin_val = [pv; 0];
            pin_var = [pvar; 0];

        case 2
            % Balance HillDeg(y) degradation against x-production terms
            cx  = (XiN(2, 2) * target_scale(2)) / col_scale(2);
            cx2 = (XiN(3, 2) * target_scale(2)) / col_scale(3);
            cx3 = (XiN(4, 2) * target_scale(2)) / col_scale(4);

            avg_prod    = mean(abs(cx*x_train + cx2*x_train.^2 + cx3*x_train.^3));
            h_basis_avg = mean(y_train ./ (hill_km + y_train));
            force_phys  = -avg_prod / max(h_basis_avg, 1e-3);

            pv   = (force_phys * col_scale(hill_deg_col)) / target_scale(2);
            s    = (target_scale(2) ./ col_scale(2:4)') .* (col_scale(hill_deg_col) / target_scale(2));
            pvar = sum(XiN_var(2:4, 2) .* s.^2);

            pin_val = [0; pv];
            pin_var = [0; pvar];

        case 3
            % No Hill term in dz/dt
            pin_val = [0; 0];
            pin_var = [0; 0];

    end
end