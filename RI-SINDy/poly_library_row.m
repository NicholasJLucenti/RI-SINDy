function row = poly_library_row(x, y, polyorder)
    row = 1;
    for k = 1:polyorder, row = [row, x^k]; end
    for k = 1:polyorder, row = [row, y^k]; end
end
