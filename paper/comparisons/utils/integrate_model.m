function Xhat = integrate_model(rhsfun, t, X0, blowup_bound, n_substeps)
% INTEGRATE_MODEL  Fixed-step RK4 integrator that reports failure as
% all-NaN instead of throwing, silently truncating, or hanging.
%
%   Xhat = INTEGRATE_MODEL(rhsfun, t, X0)
%   Xhat = INTEGRATE_MODEL(rhsfun, t, X0, blowup_bound)
%   Xhat = INTEGRATE_MODEL(rhsfun, t, X0, blowup_bound, n_substeps)
%
% rhsfun         function handle (tt, X) -> dX/dt (column vector).
% t              [nT x 1] time points to evaluate at. Must be uniformly
%                spaced (this integrator is fixed-step, not adaptive).
% X0             initial condition (row or column vector).
% blowup_bound   (optional) if any state variable's magnitude exceeds
%                this value, integration halts immediately rather than
%                continuing to compute a trajectory that's already
%                unambiguously diverging. Default 100. Goodwin and
%                NF-kB states both live in roughly the 0-15 range
%                physically (steady states are set by ratios like
%                production/decay), so 100 already gives generous
%                headroom.
% n_substeps     (optional) number of RK4 substeps taken between each
%                pair of consecutive requested points in t. Default 10.
%                Raise this for better accuracy on stiff-ish but
%                legitimately non-diverging trajectories; lower it for
%                speed if accuracy isn't a concern.
%
% WHY FIXED-STEP RK4 INSTEAD OF ode45:
% This function used to wrap ode45 with an Events-based blowup guard.
% That works fine for a trajectory that diverges gradually, but fails
% for a trajectory with a genuine finite-time blow-up (mathematically
% diverges to infinity at some finite t*, which an unregularized,
% severely collinear fit -- e.g. Traditional SINDy on NF-kB -- can
% absolutely produce). Near a true finite-time singularity, NO step
% size satisfies an adaptive solver's local error tolerance, so ode45's
% step-size controller shrinks indefinitely WITHOUT ever completing a
% step -- and since Events/OutputFcn callbacks only fire on completed
% steps, no safeguard downstream of ode45's own adaptive control can
% ever run. This looks exactly like a hang with no way to interrupt it
% short of killing MATLAB.
%
% A fixed-step integrator has no adaptive retry logic at all: every
% step costs a fixed, small, predictable amount of work, so total
% runtime is bounded by numel(t)*n_substeps RK4 stages regardless of
% how violently the solution is diverging. This is the same pattern
% already used elsewhere in this codebase for Hes1's delay-dependent
% forward integration (see forward_integrate_hes1 in hes1_comparisons.m
% and forward_integrate_nfkb in nfkb_risindy.m) -- both use a plain
% fixed-step loop rather than ode45, for the same reason.
%
% OUTPUT
%   Xhat   [nT x numel(X0)]. If any RHS evaluation throws, produces a
%          non-finite value, or any state exceeds blowup_bound at any
%          substep, Xhat is entirely NaN instead -- callers
%          (print_error_block, mc_uq_propagate) treat an all-NaN
%          trajectory as "this model diverged," not as a crash to
%          handle themselves.
%
% Only handles plain (non-delay) ODEs. Hes1's delay-dependent dynamics
% use their own Euler forward-integration loop instead -- this function
% is not a fit for history-dependent systems.
%
% See also: mc_uq_propagate.

    if nargin < 4, blowup_bound = 100; end
    if nargin < 5, n_substeps = 10; end

    nT = numel(t);
    n_dim = numel(X0);
    Xhat = NaN(nT, n_dim);

    try
        X = X0(:);
        Xhat(1,:) = X';

        for k = 1:nT-1
            dt_total = t(k+1) - t(k);
            dt_sub = dt_total / n_substeps;
            tt = t(k);

            for s = 1:n_substeps
                X = rk4_step(rhsfun, tt, X, dt_sub);
                tt = tt + dt_sub;

                if any(~isfinite(X)) || max(abs(X)) > blowup_bound
                    % Bail out immediately -- Xhat is already
                    % initialized to all-NaN, so simply returning here
                    % correctly reports "diverged" for every remaining
                    % time point without any further work.
                    return;
                end
            end

            Xhat(k+1,:) = X';
        end
    catch
        Xhat = NaN(nT, n_dim);
    end
end

function Xnext = rk4_step(rhsfun, tt, X, dt)
    k1 = rhsfun(tt,          X);
    k2 = rhsfun(tt + dt/2,   X + dt/2*k1);
    k3 = rhsfun(tt + dt/2,   X + dt/2*k2);
    k4 = rhsfun(tt + dt,     X + dt*k3);
    Xnext = X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end