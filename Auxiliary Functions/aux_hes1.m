function [pin_val, pin_var] = aux_hes1(XiN, XiN_var, vi, col_scale, target_scale, ...
                                        x_train, y_train, hill_k0, hill_n)

    if vi == 2
        pin_val = 0;
        pin_var = 0;
        return;
    end

    cx  = (XiN(2, 1) * target_scale(1)) / col_scale(2);
    cx2 = (XiN(3, 1) * target_scale(1)) / col_scale(3);

    avg_drain   = mean(cx*x_train + cx2*x_train.^2);
    h_basis_avg = mean(1 ./ (1 + (y_train./hill_k0).^hill_n));
    force_phys  = abs(avg_drain) / max(h_basis_avg, 1e-3);

    pin_val = (force_phys * col_scale(end)) / target_scale(1);

    s       = (target_scale(1) ./ col_scale(2:3)') .* (col_scale(end) / target_scale(1));
    pin_var = sum(XiN_var(2:3, 1) .* s.^2);
end