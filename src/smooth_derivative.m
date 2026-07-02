function dxdt = smooth_derivative(x, dt, sg_p, sg_f)
% SMOOTH_DERIVATIVE  Savitzky-Golay-smoothed numerical derivative.
%
%   dxdt = SMOOTH_DERIVATIVE(x, dt, sg_p, sg_f)
%
% Computes a finite-difference derivative of x (via MATLAB's gradient),
% then smooths it with a Savitzky-Golay filter to reduce the noise
% amplification that raw differentiation causes. This is the dXdt that
% feeds risindy.m's regression target -- it is NOT the identified
% model's derivative, only the numerical target derived from data.
%
% INPUTS
%   x      [N x 1]  the time series to differentiate.
%   dt     scalar sample spacing (must be uniform).
%   sg_p   Savitzky-Golay polynomial order (must be < sg_f).
%   sg_f   Savitzky-Golay frame length (must be odd).
%
% OUTPUT
%   dxdt   [N x 1]  smoothed derivative estimate.
%
% Every system so far has used sg_p=3, sg_f=11, but noisier or more
% sparsely-sampled data may need a different pair -- if the fitted
% model looks like it's chasing derivative noise, widen sg_f first.
%
% See also: risindy.

    dxdt = sgolayfilt(gradient(x, dt), sg_p, sg_f);
end