function [lo, hi] = percentile_envelope(M)
% PERCENTILE_ENVELOPE  5th/95th percentile envelope across Monte Carlo draws.
%
%   [lo, hi] = PERCENTILE_ENVELOPE(M)
%
% M    [nT x n_samples]  one row per time point, one column per draw.
%      Non-finite draws (a diverged integration) are dropped from that
%      time point's percentile computation rather than treated as
%      extreme values, so a mix of healthy and diverged draws at the
%      same time point still gives a meaningful envelope from the
%      healthy ones. If EVERY draw at a time point is non-finite, lo/hi
%      are NaN there.
%
% Computed by hand with sort() rather than prctile(), since prctile
% requires the Statistics and Machine Learning Toolbox.
%
% See also: mc_uq_propagate.

    nT = size(M, 1);
    lo = zeros(nT,1); hi = zeros(nT,1);
    sorted = sort(M, 2, 'MissingPlacement', 'last');
    nValid = sum(isfinite(sorted), 2);
    for i = 1:nT
        nv = nValid(i);
        if nv == 0
            lo(i) = NaN; hi(i) = NaN;
            continue;
        end
        li  = max(1, min(nv, round(0.05*nv)));
        hiI = max(1, min(nv, round(0.95*nv)));
        lo(i) = sorted(i, li);
        hi(i) = sorted(i, hiI);
    end
end