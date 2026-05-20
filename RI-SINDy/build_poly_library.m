function Theta = build_poly_library(varargin)

    polyorder = varargin{end};
    vars      = varargin(1:end-1);
    N         = length(vars{1});

    Theta = ones(N, 1);
    for i = 1:length(vars)
        for k = 1:polyorder
            Theta = [Theta, vars{i}.^k];
        end
    end
end