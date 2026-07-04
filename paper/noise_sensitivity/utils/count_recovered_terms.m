function [n_correct, n_spurious] = count_recovered_terms(Xi, gt_mask, threshold)
% COUNT_RECOVERED_TERMS  Classify an identified coefficient matrix
% against a known ground-truth sparsity pattern.
%
%   [n_correct, n_spurious] = COUNT_RECOVERED_TERMS(Xi, gt_mask)
%   [...] = COUNT_RECOVERED_TERMS(Xi, gt_mask, threshold)
%
% Xi         [n_lib x n_var]  identified coefficients (any units).
% gt_mask    [n_lib x n_var]  logical, true wherever the TRUE model has
%            a nonzero term at that library/equation position.
% threshold  (optional) magnitude below which a coefficient counts as
%            "not identified" for this comparison. Default 1e-6.
%
% OUTPUTS
%   n_correct   count of entries where gt_mask is true AND |Xi| > threshold.
%   n_spurious  count of entries where gt_mask is false AND |Xi| > threshold.
%
% See also: run_ensemble_sindy_3d, run_traditional_sindy_3d.

    if nargin < 3, threshold = 1e-6; end
    identified = abs(Xi) > threshold;
    n_correct  = nnz(identified & gt_mask);
    n_spurious = nnz(identified & ~gt_mask);
end