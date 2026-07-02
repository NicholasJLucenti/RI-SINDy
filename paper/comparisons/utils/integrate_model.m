function Xhat = integrate_model(rhsfun, t, X0)
% INTEGRATE_MODEL  ode45 wrapper that reports failure as all-NaN instead
% of throwing or silently truncating.
%
%   Xhat = INTEGRATE_MODEL(rhsfun, t, X0)
%
% rhsfun   function handle (tt, X) -> dX/dt (column vector).
% t        [nT x 1] time points to evaluate at.
% X0       initial condition (row or column vector).
%
% OUTPUT
%   Xhat   [nT x numel(X0)]. If ode45 throws, returns fewer time points
%          than requested, or produces any non-finite value anywhere,
%          Xhat is entirely NaN instead -- callers (print_error_block,
%          mc_uq_propagate) treat an all-NaN trajectory as "this model
%          diverged," not as a crash to handle themselves.
%
% Only handles plain (non-delay) ODEs. Hes1's delay-dependent dynamics
% use their own Euler forward-integration loop instead -- this function
% is not a fit for history-dependent systems.
%
% See also: mc_uq_propagate.

    try
        [tout, Xhat] = ode45(@(tt,X) rhsfun(tt,X), t, X0(:));
        if numel(tout) ~= numel(t) || any(~isfinite(Xhat(:)))
            Xhat = NaN(numel(t), numel(X0));
        end
    catch
        Xhat = NaN(numel(t), numel(X0));
    end
end