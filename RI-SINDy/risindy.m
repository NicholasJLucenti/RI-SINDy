function [Xi, Xi_var] = risindy(Theta, dXdt, aux_fn, pinned_col, lb, ub, eta, n_iter, prior)
    [~, n_lib] = size(Theta);
    n_var      = size(dXdt, 2);

    col_scale                       = vecnorm(Theta, 2, 1);
    col_scale(col_scale == 0)       = 1;
    ThetaN                          = Theta ./ col_scale;
    target_scale                    = vecnorm(dXdt, 2, 1);
    target_scale(target_scale == 0) = 1;
    dXdtN                           = dXdt ./ target_scale;

    free_cols   = setdiff(1:n_lib, pinned_col);
    ThetaN_free = ThetaN(:, free_cols);

    XiN     = ThetaN \ dXdtN;
    XiN_var = zeros(n_lib, n_var);
    W       = ones(numel(free_cols), n_var);

    for iter = 1:n_iter
        for vi = 1:n_var

            [pin_val, pin_var]      = aux_fn(XiN, XiN_var, vi, col_scale, target_scale);
            XiN(pinned_col, vi)     = (1-eta)*XiN(pinned_col, vi) + eta*pin_val;
            XiN_var(pinned_col, vi) = pin_var ./ col_scale(pinned_col)'.^2;

            bN = dXdtN(:, vi) - ThetaN(:, pinned_col) * XiN(pinned_col, vi);

            w       = W(:, vi);
            active  = find(w < 1e2);
            xi_free = XiN(free_cols, vi);

            if ~isempty(active)
                ThetaN_w        = ThetaN_free ./ w';
                xi_free(active) = lsqlin(ThetaN_w(:, active), bN, [], [], [], [], ...
                                         lb{vi}(active), ub{vi}(active), [], ...
                                         optimoptions('lsqlin', 'Display', 'none'));
                xi_free(w >= 1e2) = 0;
            end

            XiN(free_cols, vi) = xi_free;
            W(:, vi)           = eta*W(:, vi) + (1-eta)*compute_weights(xi_free, prior);

            resid                  = bN - ThetaN_free * xi_free;
            rv                     = var(resid) + 1e-10;
            H_mat                  = (ThetaN_free'*ThetaN_free)/rv + diag(W(:, vi));
            XiN_var(free_cols, vi) = diag(inv(H_mat));

        end
    end

    Xi     = zeros(n_lib, n_var);
    Xi_var = zeros(n_lib, n_var);
    for vi = 1:n_var
        Xi(:, vi)     = (XiN(:, vi)     * target_scale(vi)) ./ col_scale';
        Xi_var(:, vi) = (XiN_var(:, vi) * target_scale(vi)^2) ./ col_scale'.^2;
    end
end

function w = compute_weights(xi, prior)
    switch prior
        case 'Laplace'
            w = 1 ./ (abs(xi) + 1e-3);
        case 'SpikeSlab'
            v0 = 1e-4; v1 = 1.0;
            pi_incl = 1 ./ (1 + (v1/v0)*exp(-xi.^2/(2*v0)));
            w       = 1 ./ (pi_incl*v1 + (1-pi_incl)*v0);
        case 'Horseshoe'
            w = 1 ./ (0.05^2 * (xi.^2 + 1e-3));
    end
end