function Theta = build_poly_library(x, y, polyorder)
    Theta = ones(length(x), 1);
    for k = 1:polyorder, Theta = [Theta, x.^k]; end
    for k = 1:polyorder, Theta = [Theta, y.^k]; end
end
