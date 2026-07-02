function print_error_block(label, Xtrue, Xhat, stateNames)
% PRINT_ERROR_BLOCK  Print RMSE and relative L2 error per state variable.
%
%   PRINT_ERROR_BLOCK(label, Xtrue, Xhat, stateNames)
%
% Xtrue, Xhat   [nT x n_dim]  ground-truth and identified trajectories.
% stateNames    1 x n_dim cell array of state variable names.
%
% If Xhat contains any non-finite value, reports "DIVERGED" and skips
% the per-state breakdown entirely rather than printing NaN/Inf metrics.
% Works for any number of state variables -- replaces the Hes1 script's
% hardcoded 2-variable version and the Goodwin/NF-kB script's hardcoded
% 3-variable version.

    n_dim = size(Xtrue, 2);
    fprintf('[%s]\n', label);
    if any(~isfinite(Xhat(:)))
        fprintf('  DIVERGED / produced a non-finite trajectory.\n');
        return;
    end
    for d = 1:n_dim
        rmse = sqrt(mean((Xhat(:,d)-Xtrue(:,d)).^2));
        rel  = norm(Xhat(:,d)-Xtrue(:,d)) / norm(Xtrue(:,d));
        fprintf('  %-16s RMSE=%.4f  rel_L2=%.4f\n', stateNames{d}, rmse, rel);
    end
end