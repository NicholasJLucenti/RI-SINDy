function Theta = build_poly_library(varargin)
% BUILD_POLY_LIBRARY  Polynomial candidate library for any number of variables.
%
%   Theta = BUILD_POLY_LIBRARY(x1, x2, ..., xk, polyorder)
%
% Builds the polynomial portion of a SINDy library: a constant column,
% then each variable raised to powers 1..polyorder, in the order the
% variables were passed in. This is ONLY the polynomial part -- append
% any Hill/regulatory (drift-balance) columns yourself after calling
% this, e.g.:
%
%   Theta = [build_poly_library(x, y, 2), hill_func(y_delayed)];
%
% INPUTS
%   x1, x2, ... xk   column vectors, all the same length N (one state
%                    variable's time series each).
%   polyorder        highest power to include for each variable.
%
% OUTPUT
%   Theta   [N x (1 + k*polyorder)]  columns are:
%           [1, x1, x1^2, ..., x1^polyorder, x2, x2^2, ..., xk^polyorder]
%
% For the corresponding single-timestep row (used during forward
% integration), see poly_library_row.m -- keep the variable order and
% polyorder IDENTICAL between the two calls, or the fitted Xi will be
% multiplied against the wrong columns.
%
% See also: poly_library_row, risindy.

    polyorder = varargin{end};
    vars      = varargin(1:end-1);
    N         = length(vars{1});

    Theta = ones(N, 1);
    for i = 1:length(vars)
        for k = 1:polyorder
            Theta = [Theta, vars{i}.^k]; %#ok<AGROW>
        end
    end
end