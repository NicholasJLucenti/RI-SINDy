function row = poly_library_row(varargin)
% POLY_LIBRARY_ROW  Single-timestep version of build_poly_library.
%
%   row = POLY_LIBRARY_ROW(x1, x2, ..., xk, polyorder)
%
% Same column layout as build_poly_library.m, but for scalar inputs at
% one instant in time. Used inside the forward-integration loop after
% fitting, e.g.:
%
%   phi = [poly_library_row(x_prev, y_prev, polyorder), hill_func(y_tau)];
%   x_next = x_prev + dt * (phi * Xi(:,1));
%
% IMPORTANT: the variable order and polyorder here must exactly match
% whatever was passed to build_poly_library.m when Theta was built and
% fitted -- this function does not know or check that for you.
%
% See also: build_poly_library, risindy.

    polyorder = varargin{end};
    vars      = varargin(1:end-1);

    row = 1;
    for i = 1:length(vars)
        for k = 1:polyorder
            row = [row, vars{i}^k]; %#ok<AGROW>
        end
    end
end