function print_coeffs(label, Xi, names, varNames)
% PRINT_COEFFS  Print an identified coefficient table to the console.
%
%   PRINT_COEFFS(label, Xi, names)
%   PRINT_COEFFS(label, Xi, names, varNames)
%
% label     string header, e.g. 'SR3', 'Nullcline-SINDy', 'RI-SINDy'.
% Xi        [n_lib x n_var]  coefficients.
% names     1 x n_lib cell array of library term names (row labels).
% varNames  (optional) 1 x n_var cell array of equation/variable names
%           (column headers). Defaults to {'eq1','eq2',...}.
%
% Works for any number of variables -- the original Hes1 script had a
% hardcoded 2-column version and the Goodwin/NF-kB script had a
% hardcoded 3-column version; this replaces both.

    n_var = size(Xi, 2);
    if nargin < 4
        varNames = arrayfun(@(i) sprintf('eq%d', i), 1:n_var, 'UniformOutput', false);
    end
    fprintf('%s identified coefficients:\n', label);
    fprintf('%-14s', 'term');
    for v = 1:n_var, fprintf(' %10s', varNames{v}); end
    fprintf('\n');
    for i = 1:size(Xi,1)
        fprintf('%-14s', names{i});
        for v = 1:n_var, fprintf(' %10.4f', Xi(i,v)); end
        fprintf('\n');
    end
end