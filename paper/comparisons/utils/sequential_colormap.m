function cmap = sequential_colormap()
% SEQUENTIAL_COLORMAP  White-to-red colormap, no toolbox required.
%
%   cmap = SEQUENTIAL_COLORMAP()
%
% For effectively one-sided data (e.g. a correlation matrix where every
% pair is positively correlated) -- a diverging colormap would waste
% half its range on negative values that never occur and crush the real
% variation into a sliver.
%
% See also: diverging_colormap, plot_collinearity_heatmap.
    n = 256;
    cmap = [ones(n,1), linspace(1,0,n)', linspace(1,0,n)'];
end