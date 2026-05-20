function row = poly_library_row(varargin)
    polyorder = varargin{end};
    vars      = varargin(1:end-1);

    row = 1;
    for i = 1:length(vars)
        for k = 1:polyorder
            row = [row, vars{i}^k];
        end
    end
end