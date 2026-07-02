function plot_collinearity_heatmap(Theta, names, titleSuffix)
% PLOT_COLLINEARITY_HEATMAP  Pairwise correlation heatmap of every
% candidate library column, auto-switching between a diverging
% (blue-white-red) and sequential (white-red) colormap depending on
% whether the correlations present are genuinely two-sided or
% effectively one-sided.
%
%   PLOT_COLLINEARITY_HEATMAP(Theta, names, titleSuffix)
%
% Theta        [N x n_lib]  the EXACT library matrix used for fitting,
%              evaluated on the exact data every method is actually fit
%              on. Drop the constant column yourself before calling --
%              a zero-variance column has an undefined correlation and
%              will corrupt the heatmap if left in.
% names        1 x n_lib cell array of column labels for the axes
%              (same length as Theta has columns, so also without the
%              constant column).
% titleSuffix  string used in the figure name and title, e.g. 'hes1' or
%              'goodwin'.
%
% See also: diverging_colormap, sequential_colormap.

    R = corrcoef(Theta);
    offDiag = R(~eye(size(R)));
    loval = min(offDiag); hival = max(offDiag);

    figure('Name', sprintf('%s library collinearity', titleSuffix), 'Color','w');
    imagesc(R);
    if loval < -0.05
        % Genuinely two-sided: keep correlation = 0 anchored at white,
        % tighten the symmetric bounds to the largest magnitude actually
        % present rather than wasting range on +/-1 if never reached.
        maxAbs = max(abs([loval, hival]));
        caxis([-maxAbs, maxAbs]);
        colormap(diverging_colormap());
    else
        % Effectively one-sided: a diverging map would collapse almost
        % all the real variation into one visual band. Use a sequential
        % map and tighten bounds to the observed range for contrast.
        pad = 0.02*(hival-loval); if pad==0, pad=0.02; end
        caxis([max(loval-pad,-1), min(hival+pad,1)]);
        colormap(sequential_colormap());
    end
    colorbar;
    set(gca, 'XTick', 1:numel(names), 'XTickLabel', names, 'XTickLabelRotation', 45, ...
             'YTick', 1:numel(names), 'YTickLabel', names, 'FontSize', 8);
    title(sprintf('Candidate Library Collinearity -- %s', upper(titleSuffix)));
    axis square;
end