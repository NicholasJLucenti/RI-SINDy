function row = poly_library_row(varargin)

% Same column layout as build_poly_library.m, but for scalar inputs at
% one instant in time. Used inside the forward-integration loop after fitting

    polyorder = varargin{end};
    vars      = varargin(1:end-1);

    row = 1;
    for i = 1:length(vars)
        for k = 1:polyorder
            row = [row, vars{i}^k]; %#ok<AGROW>
        end
    end
end