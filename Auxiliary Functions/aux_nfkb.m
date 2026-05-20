function [pin_val, pin_var] = aux_NFkB(XiN, XiN_var, ~, col_scale, target_scale, ...
                                        x_train, z_train, hill_k0, hill_n, mode)
% aux_nfkb — pinned Hill auxiliary for the NF-kB network.
%   mode='xy' : called from risindy on [dxdt, dydt]; vi=1,2 -> no Hill
%   mode='z'  : called from risindy on dzdt alone;  vi=1   -> HillAct(x)

    if strcmp(mode, 'xy')
        % Equations 1 & 2 have no Hill term
        pin_val = 0;
        pin_var = 0;
        return;
    end

    % mode='z': single equation, XiN is [10x1], vi is always 1
    % Balance HillAct(x) production against z drain terms (cols 6,7,8)
    cz  = (XiN(6, 1) * target_scale(1)) / col_scale(6);
    cz2 = (XiN(7, 1) * target_scale(1)) / col_scale(7);
    cz3 = (XiN(8, 1) * target_scale(1)) / col_scale(8);

    avg_drain   = mean(abs(cz*z_train + cz2*z_train.^2 + cz3*z_train.^3));
    h_basis_avg = mean(x_train.^hill_n ./ (hill_k0^hill_n + x_train.^hill_n));
    force_phys  = avg_drain / max(h_basis_avg, 1e-3);

    pin_val = (force_phys * col_scale(end)) / target_scale(1);
    s       = (target_scale(1) ./ col_scale(6:8)') .* (col_scale(end) / target_scale(1));
    pin_var = sum(XiN_var(6:8, 1) .* s.^2);
end