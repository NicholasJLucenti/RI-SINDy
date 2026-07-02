function [Xi, Xi_var, diagnostics] = risindy(Theta, dXdt, drift_fn, pinned_col, lb, ub, opts)
% RISINDY  Regulation-Informed Sparse Identification of Nonlinear Dynamics.
%
%   [Xi, Xi_var] = RISINDY(Theta, dXdt, drift_fn, pinned_col, lb, ub)
%   [Xi, Xi_var] = RISINDY(Theta, dXdt, drift_fn, pinned_col, lb, ub, opts)
%   [Xi, Xi_var, diagnostics] = RISINDY(...)
%
% Fits a sparse dynamical model  dXdt ~ Theta * Xi  using iterative
% reweighted sparse regression under a Bayesian shrinkage prior, while
% one or more reserved library columns ("drift-balance" / regulatory
% terms -- e.g. Hill functions) are NEVER entered into the regression.
% Instead, on every pass their coefficients are pinned to a value
% returned by a user-supplied drift_fn, which computes the value that
% balances the currently-estimated drain/production terms against the
% regulatory term's typical basis magnitude. This is what distinguishes
% RI-SINDy from vanilla sparse regression: the regulatory term is
% supplied from independent physical/regulatory information rather than
% discovered (or discarded) by the sparse solver.
%
% See drift_balance_template.m for the full contract a drift_fn must
% satisfy, plus a worked minimal example.
%
% INPUTS
%   Theta       [N x n_lib]  library matrix (ALL columns, including
%               pinned/regulatory ones). Build the polynomial part with
%               build_poly_library.m, then append Hill/regulatory
%               columns by hand.
%   dXdt        [N x n_var]  numerical state derivatives (one column
%               per state variable / ODE equation). Use
%               smooth_derivative.m to compute these.
%   drift_fn    function handle:
%                   [pin_val, pin_var] = drift_fn(XiN_smooth, XiN_var, vi, col_scale, target_scale)
%               Called once per equation (vi) per iteration. Must
%               return pin_val, pin_var as column vectors the same
%               length as pinned_col (use 0 for equations where a given
%               regulatory term is inactive). Both must already be in
%               NORMALIZED (XiN) space -- see drift_balance_template.m.
%   pinned_col  vector of column indices into Theta reserved for
%               drift-balance terms. These columns are never entered
%               into the sparse regression; they are always overwritten
%               by drift_fn on every pass.
%   lb, ub      1 x n_var cell arrays. lb{vi}, ub{vi} are lower/upper
%               bound vectors for the FREE (non-pinned) columns of
%               equation vi, in the order setdiff(1:n_lib, pinned_col).
%               Use +-inf freely; use tight bounds (e.g. [0,0]) to force
%               a term to zero when you know its sign/absence a priori.
%   opts        (optional) struct of hyperparameters. Any field you
%               omit falls back to a default -- see local function
%               default_risindy_opts for the full list and meanings.
%               Fields: eta_pin, eta_weight, eta_drain, n_iter,
%               w_threshold, var_floor, prior, prior_params.
%
% OUTPUTS
%   Xi          [n_lib x n_var]  identified coefficients, PHYSICAL units.
%   Xi_var      [n_lib x n_var]  posterior variance of Xi, PHYSICAL units.
%   diagnostics struct with the normalized-space fit (XiN, XiN_var,
%               XiN_smooth, W, free_cols, pinned_col) -- useful for
%               debugging convergence or inspecting the drain-smoothed
%               trajectory that fed the drift-balance calculation.
%
% See also: build_poly_library, poly_library_row, get_delayed,
%           smooth_derivative, drift_balance_template.

    if nargin < 7, opts = struct(); end
    opts = default_risindy_opts(opts);

    n_lib = size(Theta, 2);
    n_var = size(dXdt, 2);

    %% --- normalization ---
    col_scale                       = vecnorm(Theta, 2, 1);
    col_scale(col_scale == 0)       = 1;
    ThetaN                          = Theta ./ col_scale;
    target_scale                    = vecnorm(dXdt, 2, 1);
    target_scale(target_scale == 0) = 1;
    dXdtN                           = dXdt ./ target_scale;

    free_cols   = setdiff(1:n_lib, pinned_col);
    ThetaN_free = ThetaN(:, free_cols);

    %% --- initialize: joint LS informed by the full library, then discard
    %      the pinned rows (those are handled by drift_fn, never by LS) ---
    XiN_full_ls        = ThetaN \ dXdtN;
    XiN                = zeros(n_lib, n_var);
    XiN(free_cols, :)  = XiN_full_ls(free_cols, :);
    XiN_var            = zeros(n_lib, n_var);
    XiN_smooth         = XiN;   % drain-smoothed copy fed to drift_fn each pass
    W                  = ones(numel(free_cols), n_var);

    for iter = 1:opts.n_iter
        for vi = 1:n_var

            % --- pin drift-balance columns using the SMOOTHED drain estimate ---
            [pin_val, pin_var] = drift_fn(XiN_smooth, XiN_var, vi, col_scale, target_scale);
            XiN(pinned_col, vi)     = (1 - opts.eta_pin) * XiN(pinned_col, vi) + opts.eta_pin * pin_val;
            XiN_var(pinned_col, vi) = pin_var;   % already normalized-space; do NOT rescale by col_scale again

            % --- sparse regression on the remaining (free) columns ---
            bN      = dXdtN(:, vi) - ThetaN(:, pinned_col) * XiN(pinned_col, vi);
            w       = W(:, vi);
            active  = find(w < opts.w_threshold);
            xi_free = XiN(free_cols, vi);

            if ~isempty(active)
                ThetaN_w        = ThetaN_free ./ w';
                xi_free(active) = lsqlin(ThetaN_w(:, active), bN, [], [], [], [], ...
                                          lb{vi}(active), ub{vi}(active), [], ...
                                          optimoptions('lsqlin', 'Display', 'none'));
                xi_free(w >= opts.w_threshold) = 0;
            end
            XiN(free_cols, vi) = xi_free;

            % --- reweight sparsity prior ---
            new_w    = compute_weights(xi_free, opts.prior, opts.prior_params);
            W(:, vi) = opts.eta_weight * W(:, vi) + (1 - opts.eta_weight) * new_w;

            % --- posterior variance on free columns ---
            resid                  = bN - ThetaN_free * xi_free;
            rv                     = var(resid) + opts.var_floor;
            H_mat                  = (ThetaN_free' * ThetaN_free) / rv + diag(W(:, vi));
            XiN_var(free_cols, vi) = diag(inv(H_mat));
        end

        % --- smooth the drain (free-column) coefficients for the NEXT
        %     iteration's drift-balance calculation. This stabilizes
        %     drift_fn against noisy per-iteration swings in the
        %     regression fit; it does NOT affect XiN_var. ---
        XiN_smooth = opts.eta_drain * XiN_smooth + (1 - opts.eta_drain) * XiN;
    end

    %% --- denormalize to physical units ---
    Xi     = zeros(n_lib, n_var);
    Xi_var = zeros(n_lib, n_var);
    for vi = 1:n_var
        Xi(:, vi)     = (XiN(:, vi)     * target_scale(vi))   ./ col_scale';
        Xi_var(:, vi) = (XiN_var(:, vi) * target_scale(vi)^2) ./ col_scale'.^2;
    end

    diagnostics = struct('XiN', XiN, 'XiN_var', XiN_var, 'XiN_smooth', XiN_smooth, ...
                          'W', W, 'free_cols', free_cols, 'pinned_col', pinned_col, ...
                          'col_scale', col_scale, 'target_scale', target_scale);
end

function w = compute_weights(xi, prior, p)
    switch prior
        case 'Laplace'
            w = 1 ./ (abs(xi) + p.eps_sparsity);
        case 'SpikeSlab'
            pi_incl = 1 ./ (1 + (p.v1/p.v0) * exp(-xi.^2 / (2*p.v0)));
            w       = 1 ./ (pi_incl*p.v1 + (1-pi_incl)*p.v0);
        case 'Horseshoe'
            w = 1 ./ (p.tau0^2 * (xi.^2 + p.eps_sparsity));
        otherwise
            error('risindy:unknownPrior', ...
                  'Unknown prior "%s". Use ''Laplace'', ''SpikeSlab'', or ''Horseshoe''.', prior);
    end
end

function opts = default_risindy_opts(opts)
% Defaults are a reasonable starting point only. Every system we've run
% so far (Hes1, Goodwin, NF-kB) has needed at least one of these tuned
% away from default -- treat opts as required reading per new system,
% not as something to leave untouched.
    if ~isfield(opts, 'eta_pin'),     opts.eta_pin     = 0.4;   end  % blend rate for the drift-balance pin value (share given to the NEW estimate)
    if ~isfield(opts, 'eta_weight'),  opts.eta_weight  = 0.7;   end  % blend rate for sparsity weights (share RETAINED from the OLD estimate)
    if ~isfield(opts, 'eta_drain'),   opts.eta_drain   = 0.7;   end  % blend rate for drain-coefficient smoothing (share RETAINED from the OLD estimate)
    if ~isfield(opts, 'n_iter'),      opts.n_iter      = 20;    end
    if ~isfield(opts, 'w_threshold'), opts.w_threshold = 1e2;   end  % a free column is still "active" (eligible for lsqlin) while its weight is below this
    if ~isfield(opts, 'var_floor'),   opts.var_floor   = 1e-10; end
    if ~isfield(opts, 'prior'),       opts.prior       = 'SpikeSlab'; end
    if ~isfield(opts, 'prior_params')
        switch opts.prior
            case 'Laplace';   opts.prior_params = struct('eps_sparsity', 1e-3);
            case 'SpikeSlab'; opts.prior_params = struct('v0', 1e-4, 'v1', 1.0);
            case 'Horseshoe'; opts.prior_params = struct('tau0', 0.05, 'eps_sparsity', 1e-3);
        end
    end
end