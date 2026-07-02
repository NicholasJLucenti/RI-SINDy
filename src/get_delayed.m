function y_delayed = get_delayed(t, y, tau)
% GET_DELAYED  Linearly-interpolated delayed version of a time series.
%
%   y_delayed = GET_DELAYED(t, y, tau)
%
% Returns y(t - tau) at every sample time in t, via linear
% interpolation, clamped so no query time goes below t(1). Use this to
% build delayed-argument regulatory terms, e.g. a Hill function acting
% on a protein concentration tau hours in the past:
%
%   y_tau     = get_delayed(t, y, tau);
%   HillDelay = hill_func(y_tau);
%
% INPUTS
%   t     [N x 1]  sample times (must be sorted, evenly or unevenly spaced).
%   y     [N x 1]  the series to delay.
%   tau   scalar delay (same time units as t). tau = 0 returns y unchanged.
%
% OUTPUT
%   y_delayed   [N x 1]  y sampled at (t - tau), clamped at t(1).
%
% See also: risindy.

    y_delayed = interp1(t, y, max(t - tau, 0), 'linear');
end