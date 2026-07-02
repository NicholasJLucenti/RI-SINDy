function [lo, hi] = mc_uq_propagate(draw_fn, Xi, Xi_var, n_samples, seed)
% MC_UQ_PROPAGATE  Monte Carlo propagation of a coefficient posterior
% into a trajectory uncertainty band.
%
%   [lo, hi] = MC_UQ_PROPAGATE(draw_fn, Xi, Xi_var, n_samples)
%   [lo, hi] = MC_UQ_PROPAGATE(draw_fn, Xi, Xi_var, n_samples, seed)
%
% Draws n_samples coefficient sets from N(Xi, Xi_var) (elementwise,
% diagonal covariance -- no cross-term covariance is modeled), calls
% draw_fn once per draw to forward-integrate that sample into a full
% trajectory, then returns the 5th/95th percentile envelope across
% draws at every time point, per state dimension.
%
% INPUTS
%   draw_fn     function handle: Xhat = draw_fn(Xi_sample). Xi_sample is
%               the same size as Xi; Xhat must be [nT x n_dim]. All
%               system-specific integration logic (plain ode45, or
%               delay-aware stepping for a history-dependent system like
%               Hes1) lives inside this closure -- MC_UQ_PROPAGATE
%               itself is completely system-agnostic. Called once extra
%               up front (on the unperturbed Xi) just to learn the
%               trajectory shape.
%   Xi, Xi_var  [n_lib x n_var]  identified coefficients and their
%               PHYSICAL-unit posterior variance.
%   n_samples   number of Monte Carlo draws.
%   seed        (optional) rng seed, default 42, for reproducibility.
%
% OUTPUTS
%   lo, hi      [nT x n_dim]  5th/95th percentile envelope.
%
% See also: percentile_envelope, integrate_model.

    if nargin < 5, seed = 42; end
    rng(seed);

    Xhat0 = draw_fn(Xi);
    nT = size(Xhat0, 1);  n_dim = size(Xhat0, 2);

    Xmc = NaN(nT, n_dim, n_samples);
    for s = 1:n_samples
        Xi_s = Xi + sqrt(max(Xi_var, 0)) .* randn(size(Xi));
        Xmc(:, :, s) = draw_fn(Xi_s);
    end

    lo = zeros(nT, n_dim); hi = zeros(nT, n_dim);
    for d = 1:n_dim
        [lo(:,d), hi(:,d)] = percentile_envelope(squeeze(Xmc(:,d,:)));
    end
end