function [pin_val, pin_var] = drift_balance_template(XiN_smooth, XiN_var, vi, col_scale, target_scale, varargin)
% DRIFT_BALANCE_TEMPLATE  Starting point for writing a new system's
% drift-balance (formerly "auxiliary"/"hyperstate") function.
%
% THIS FILE DOES NOT RUN ON ITS OWN. Copy it, rename it
% (drift_balance_<yoursystem>.m), and fill in the marked sections. It
% exists purely to document the contract risindy.m expects, since that
% contract lives in normalized space and is easy to get wrong.
%
% ============================== THE IDEA =================================
% RI-SINDy reserves one or more library columns for regulatory terms
% (e.g. Hill functions) and refuses to let sparse regression touch them.
% Instead, each pass, this function computes what that regulatory term's
% coefficient MUST be for the equation to balance: it looks at how large
% the (already-fitted) drain/production terms currently are, looks at
% how large the regulatory term's own basis function typically is over
% the data, and divides one by the other. That's the whole idea --
% "drift balance" means solving for the coefficient that makes the known
% physical force balance the identified polynomial drift.
%
% ============================ THE CONTRACT ================================
% risindy.m calls this function once per equation (vi) per iteration as:
%
%   [pin_val, pin_var] = your_fn(XiN_smooth, XiN_var, vi, col_scale, target_scale, <your extra args>)
%
% XiN_smooth    [n_lib x n_var]  the CURRENT, EMA-SMOOTHED estimate of
%               all coefficients in NORMALIZED space (this is XiN, not
%               Xi -- see below). Use this, not XiN_var, to read the
%               current drain/production coefficients you're balancing
%               against. It's smoothed on purpose so your drift-balance
%               calculation isn't jerked around by a single noisy
%               regression pass.
% XiN_var       [n_lib x n_var]  variance of the RAW (unsmoothed)
%               coefficients, same normalized space. Use this only when
%               propagating uncertainty into pin_var.
% vi            which equation (which column of dXdt) is being fit right
%               now. Your function will typically have a switch/if on vi
%               -- e.g. only equation 1 has this Hill term active, so
%               return zero for every other vi.
% col_scale     [1 x n_lib]  vecnorm of each RAW Theta column. This is
%               what converts between normalized and physical space.
% target_scale  [1 x n_var]  vecnorm of each RAW dXdt column. Same role,
%               for the target/output side.
%
% RETURN VALUES -- both MUST already be in NORMALIZED (XiN) space:
%   pin_val   column vector, length = numel(pinned_col), the value(s) to
%             write into XiN(pinned_col, vi) this pass.
%   pin_var   column vector, same length, the corresponding variance(s)
%             to write into XiN_var(pinned_col, vi).
%
% To convert a PHYSICAL coefficient/value to the NORMALIZED space these
% outputs need to be in, multiply by col_scale(that column) and divide
% by target_scale(vi):
%
%   pin_val_normalized = physical_value * col_scale(pinned_col) / target_scale(vi);
%
% Do NOT apply any further division by col_scale after your function
% returns -- risindy.m writes pin_val / pin_var straight into
% XiN / XiN_var with no additional rescaling. (An earlier version of
% this codebase divided pin_var by col_scale(pinned_col)^2 a second time
% at the call site -- that was a bug that silently distorted the
% reported uncertainty on the regulatory coefficient. Don't reintroduce it.)
%
% ============================ WORKED EXAMPLE ===============================
% NOTE: drift_balance_generic.m now exists and does exactly this
% calculation for you -- in practice, prefer calling that over hand-
% deriving the math below. See drift_balance_hes1.m for a real example:
% it's a ~15-line file that just builds drain_cols/pinned_col/
% drain_basis/reg_basis and calls drift_balance_generic.m once. The
% derivation below is kept for reference/teaching, and for the rare case
% where a system's balance condition doesn't fit that generic pattern.
%
% Minimal single-variable self-repressing system:
%     dx/dt = -k*x + alpha * Hill(x),   Hill(x) = 1 / (1 + (x/k0)^n)
%
% Theta   = [build_poly_library(x, 1), hill_func(x)];   % columns: [1, x, Hill(x)]
% pinned_col = 3;                                        % Hill(x) is column 3
% lb{1} = [0, -inf];   ub{1} = [0, 0];                    % force const=0, x coeff <= 0
%
% Equivalent one-liner using drift_balance_generic.m:
%   [pin_val, pin_var] = drift_balance_generic(XiN_smooth, XiN_var, vi, ...
%                             col_scale, target_scale, 2, 3, x_data, hill_func(x_data,hill_k0,hill_n));
%
% Or, spelled out by hand (what drift_balance_generic.m does internally):
%
% function [pin_val, pin_var] = drift_balance_toy(XiN_smooth, XiN_var, vi, col_scale, target_scale, x_data, hill_k0, hill_n)
%     % physical-space drain coefficient currently estimated for the x term (column 2)
%     cx = (XiN_smooth(2, vi) * target_scale(vi)) / col_scale(2);
%
%     avg_drain   = mean(abs(cx * x_data));                       % typical physical drain magnitude
%     h_basis_avg = mean(1 ./ (1 + (x_data ./ hill_k0).^hill_n));  % typical Hill basis magnitude
%     force_phys  = avg_drain / max(h_basis_avg, 1e-3);           % physical Hill coefficient that balances it
%
%     pin_val = (force_phys * col_scale(3)) / target_scale(vi);   % -> normalized space
%
%     scaling_vec = col_scale(3) / col_scale(2);                  % sensitivity of pin_val to XiN(2,vi)
%     pin_var     = XiN_var(2, vi) * scaling_vec^2;                % propagate variance, already normalized
% end
%
% ============================================================================

    error('drift_balance_template:notImplemented', ...
          ['This is a template, not a runnable function. Copy this file to ' ...
           'drift_balance_<yoursystem>.m and implement the marked sections.']);
end