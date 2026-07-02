function cmap = diverging_colormap()
% DIVERGING_COLORMAP  Blue-white-red colormap, no toolbox required.
%
%   cmap = DIVERGING_COLORMAP()
%
% For genuinely two-sided data centered at zero (e.g. a correlation
% matrix with both positive and negative entries). See also
% sequential_colormap.m for the one-sided case.
%
% See also: sequential_colormap, plot_collinearity_heatmap.
    n = 256;
    half = floor(n/2);
    blue_to_white = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1)];
    white_to_red  = [ones(n-half,1), linspace(1,0,n-half)', linspace(1,0,n-half)'];
    cmap = [blue_to_white; white_to_red];
end