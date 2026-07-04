function plot_metric_heatmap(M, rowLabels, colLabels, titleStr, colorMode, denom, climits)
% PLOT_METRIC_HEATMAP  Single metric heatmap with NaN cells shown as
% "div" (diverged / not reported), used for the noise-sensitivity
% figure. Extracted from noise_sensitivity_heatmap.m so both the
% saved-numbers display script and the live noise_sensitivity_sweep.m
% script can call the same plotting code.
%
%   PLOT_METRIC_HEATMAP(M, rowLabels, colLabels, titleStr, colorMode)
%   PLOT_METRIC_HEATMAP(M, rowLabels, colLabels, titleStr, colorMode, denom, climits)
%
% M           matrix of values to display as a heatmap. NaN entries are
%             rendered as light gray "div" cells.
% rowLabels   cell array of row labels (e.g. noise levels).
% colLabels   cell array of column labels (e.g. method names).
% titleStr    subplot title.
% colorMode   'blue_white'   = white (low) to blue (high)
%             'orange_white' = white (low) to dark orange (high)
% denom       (optional) if non-empty, annotate each cell as "value/denom".
% climits     (optional) [vmin, vmax] to manually set the color scale
%             range; omit or pass [] for automatic scaling from the data.
%
% See also: noise_sensitivity_heatmap, noise_sensitivity_sweep.

    [nRows, nCols] = size(M);
    validMask = ~isnan(M);

    % Background for NaN (diverged / not reported) cells
    bg = ones(nRows, nCols, 3) * 0.85;   % light gray everywhere by default

    % Determine colorbar limits
    if nargin >= 7 && ~isempty(climits)
        vmin = climits(1);
        vmax = climits(2);
    else
        if any(validMask(:))
            vmin = min(M(validMask));
            vmax = max(M(validMask));
            if vmax == vmin, vmax = vmin + 1; end
        else
            vmin = 0;
            vmax = 1;
        end
    end

    % Normalize data mapping based on limits
    M_clamped = M;
    M_clamped(M_clamped < vmin) = vmin;
    M_clamped(M_clamped > vmax) = vmax;
    Mnorm = (M_clamped - vmin) / (vmax - vmin);

    for i = 1:nRows
        for j = 1:nCols
            if ~validMask(i,j)
                continue;   % leave as light gray
            end
            t = Mnorm(i,j);
            switch colorMode
                case 'blue_white'
                    % white [1,1,1] at low (t=0) -> blue [0,0,1] at high (t=1)
                    bg(i,j,:) = [1-t, 1-t, 1];
                case 'orange_white'
                    % white [1,1,1] at low (t=0) -> dark orange [0.8,0.4,0] at high (t=1)
                    bg(i,j,:) = [1 - 0.2*t, 1 - 0.6*t, 1-t];
            end
        end
    end

    image(bg);
    axis image;
    ax = gca;
    set(ax, 'XTick', 1:nCols, 'XTickLabel', colLabels, ...
            'YTick', 1:nRows, 'YTickLabel', rowLabels, ...
            'FontSize', 8, 'TickLength', [0 0]);
    title(titleStr, 'FontSize', 9, 'FontWeight','normal');
    xlabel('Method', 'FontSize', 8);
    ylabel('Noise level', 'FontSize', 8);

    % Grid lines between cells
    hold on;
    for k = 0.5:1:nCols+0.5
        plot([k k], [0.5 nRows+0.5], 'Color',[0.6 0.6 0.6], 'LineWidth',0.5);
    end
    for k = 0.5:1:nRows+0.5
        plot([0.5 nCols+0.5], [k k], 'Color',[0.6 0.6 0.6], 'LineWidth',0.5);
    end

    % Cell annotations
    for i = 1:nRows
        for j = 1:nCols
            if ~validMask(i,j)
                txt = 'div';
                col = [0.4 0.4 0.4];
            else
                v = M(i,j);
                if ~isempty(denom)
                    txt = sprintf('%d/%d', round(v), denom);
                elseif abs(v - round(v)) < 1e-9
                    txt = sprintf('%d', round(v));
                else
                    txt = sprintf('%.3f', v);
                end
                col = [0.05 0.05 0.05];
            end
            text(j, i, txt, 'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', 'FontSize', 8, 'Color', col);
        end
    end

    %% Add matching colorbar
    cbSteps = 256;
    t_vals = linspace(0, 1, cbSteps)';
    customMap = zeros(cbSteps, 3);

    switch colorMode
        case 'blue_white'
            customMap(:, 1) = 1 - t_vals;
            customMap(:, 2) = 1 - t_vals;
            customMap(:, 3) = 1;
        case 'orange_white'
            customMap(:, 1) = 1 - 0.2 * t_vals;
            customMap(:, 2) = 1 - 0.6 * t_vals;
            customMap(:, 3) = 1 - t_vals;
    end

    colormap(ax, customMap);
    cb = colorbar(ax, 'FontSize', 8);
    set(ax, 'CLim', [vmin, vmax]);

    % Customize ticks based on rules specified
    if ~isempty(denom)
        ylabel(cb, 'Count', 'FontSize', 8);
        set(cb, 'Ticks', 0:denom);
    elseif nargin >= 7 && ~isempty(climits) && climits(1) == 0 && climits(2) == 0.6
        set(cb, 'Ticks', 0:0.1:0.6);
    else
        if floor(vmin) == vmin && floor(vmax) == vmax && (vmax - vmin) <= 15
            set(cb, 'Ticks', vmin:ceil((vmax-vmin)/5):vmax);
        end
    end
end