function Xhat = integrate_model(rhsfun, t, X0, blowup_bound)
% INTEGRATE_MODEL  ode45 wrapper that reports failure as all-NaN instead
% of throwing or silently truncating.
%
%   Xhat = INTEGRATE_MODEL(rhsfun, t, X0)
%   Xhat = INTEGRATE_MODEL(rhsfun, t, X0, blowup_bound)
%
% rhsfun         function handle (tt, X) -> dX/dt (column vector).
% t              [nT x 1] time points to evaluate at.
% X0             initial condition (row or column vector).
% blowup_bound   (optional) if any state variable's magnitude exceeds
%                this value, integration halts immediately via an
%                ode45 event rather than continuing to grind through
%                ever-smaller step sizes trying to numerically resolve
%                a trajectory that's already unambiguously diverging.
%                Default 1e6. Raise this if a system's physical units
%                are legitimately large and 1e6 is too tight a leash;
%                lower it to catch divergence (and give up) faster.
%
% OUTPUT
%   Xhat   [nT x numel(X0)]. If ode45 throws, hits the blowup_bound
%          event, returns fewer time points than requested, or produces
%          any non-finite value anywhere, Xhat is entirely NaN instead
%          -- callers (print_error_block, mc_uq_propagate) treat an
%          all-NaN trajectory as "this model diverged," not as a crash
%          to handle themselves.
%
% Only handles plain (non-delay) ODEs. Hes1's delay-dependent dynamics
% use their own Euler forward-integration loop instead -- this function
% is not a fit for history-dependent systems.
%
% See also: mc_uq_propagate.

    if nargin < 4, blowup_bound = 1e6; end

    try
        warnState = warning('off', 'MATLAB:ode45:IntegrationTolNotMet');
        cleanupObj = onCleanup(@() warning(warnState));   % restores the warning's on/off state after this call, even if it errors

        odeOpts = odeset('Events', @(tt,X) blowup_event(tt, X, blowup_bound));
        [tout, Xhat] = ode45(@(tt,X) rhsfun(tt,X), t, X0(:), odeOpts);
        if numel(tout) ~= numel(t) || any(~isfinite(Xhat(:)))
            Xhat = NaN(numel(t), numel(X0));
        end
    catch
        Xhat = NaN(numel(t), numel(X0));
    end
end

function [value, isterminal, direction] = blowup_event(~, X, blowup_bound)
    % Fires (and halts integration) the instant any state's magnitude
    % crosses blowup_bound, rather than letting ode45 keep shrinking its
    % step size trying to resolve a trajectory that's already diverging.
    value      = blowup_bound - max(abs(X));   % crosses zero when any |X(i)| exceeds blowup_bound
    isterminal = 1;                             % stop integration when triggered
    direction  = -1;                            % only trigger while value is decreasing (approaching/crossing zero from above)
end