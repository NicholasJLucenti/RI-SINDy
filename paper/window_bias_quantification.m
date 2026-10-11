%% window_length_bias_figure.m
%
% Illustrates how the length and starting phase of an observation window
% affect the drift-balance estimate of a regulatory coefficient in a
% bounded oscillatory system (Goodwin-type limit cycle).
%
% Generates Panel 1 (multi-trajectory) and Panel 2 (heatmap) side-by-side.

clear; clc; close all;

%% ---------------- Ground-truth system parameters ----------------------
a1 = 1.0;         % x1 decay rate
a2 = 1.0;         % x2 decay rate
a3 = 1.0;         % x3 decay rate
b  = 1.0;         % x1 -> x2 production rate
c  = 1.0;         % x2 -> x3 production rate
n  = 10;          % Hill coefficient (steep negative feedback)
xi_R_true = 15.0; % regulatory coefficient

H = @(x3) 1 ./ (1 + x3.^n);
odefun = @(t, x) [ xi_R_true*H(x(3)) - a1*x(1);
                   b*x(1) - a2*x(2);
                   c*x(2) - a3*x(3) ];

%% ---------------- Simulate to steady-state limit cycle -----------------
x0 = [2.0; 2.0; 2.0];
tspan_burn = [0 400]; % Burn-in for limit cycle to settle
opts = odeset('RelTol',1e-10,'AbsTol',1e-12);
[~, Xburn] = ode45(odefun, tspan_burn, x0, opts);
x0_ss = Xburn(end,:)';   

dt = 0.01;
T_total = 100;
tt = 0:dt:T_total;
[~, X] = ode45(odefun, tt, x0_ss, opts);
x1 = X(:,1); x3 = X(:,3);
Hx3 = H(x3);

%% ---------------- Estimate the oscillation period -----------------------
[~, locs] = findpeaks(x1, tt, 'MinPeakProminence', (max(x1)-min(x1))*0.1);
period = mean(diff(locs));
fprintf('Estimated period: %.3f time units (%d peaks found)\n', period, numel(locs));

%% ---------------- Sweep 1: window length, fixed start phases -----------
n_lengths   = 60;
window_lengths_cycles = linspace(0, 6, n_lengths);          % linear spacing
start_phases_frac     = [0.0 0.15 0.30 0.45 0.60 0.75];         % fraction of period
t0_ref = 20;  % reference start time

est_by_phase = nan(numel(start_phases_frac), n_lengths);
for ip = 1:numel(start_phases_frac)
    t0 = t0_ref + start_phases_frac(ip)*period;
    for il = 1:n_lengths
        T = window_lengths_cycles(il) * period;
        idx = (tt >= t0) & (tt <= t0 + T);
        if nnz(idx) < 5, continue; end
        est_by_phase(ip, il) = a1 * mean(x1(idx)) / mean(Hx3(idx));
    end
end

%% ---------------- Sweep 2: full 2D bias map (length x start phase) -----
n_len2   = 50;
n_phase2 = 50;
window_lengths2 = linspace(0.15, 4, n_len2);                 % cycles, linearly spaced
start_phases2   = linspace(0, 1, n_phase2);                 % fraction of a cycle

bias_map = nan(n_phase2, n_len2);
for ip = 1:n_phase2
    t0 = t0_ref + start_phases2(ip)*period;
    for il = 1:n_len2
        T = window_lengths2(il) * period;
        idx = (tt >= t0) & (tt <= t0 + T);
        if nnz(idx) < 5, continue; end
        est = a1 * mean(x1(idx)) / mean(Hx3(idx));
        bias_map(ip, il) = 100 * (est - xi_R_true) / xi_R_true;
    end
end

%% ======================= FIGURE ========================================
fig = figure('Color','w','Position',[80 200 1200 500]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing','compact', 'Padding','compact');

cmap_phase = turbo(numel(start_phases_frac));

%% --- Left Panel: estimate vs window length (Linear X-scale) -----------
ax1 = nexttile(tl, 1);
hold(ax1,'on'); box(ax1,'on');
yline(ax1, xi_R_true, 'k--', 'LineWidth', 1.5);
leg_h = gobjects(numel(start_phases_frac),1);
for ip = 1:numel(start_phases_frac)
    leg_h(ip) = plot(ax1, window_lengths_cycles, est_by_phase(ip,:), '-', ...
        'Color', cmap_phase(ip,:), 'LineWidth', 1.8);
end
xline(ax1, 2, ':', 'Color',[0.3 0.3 0.3], 'LineWidth', 1.2);

% Set X-axis scale to linear with clear tick increments
set(ax1, 'XScale', 'linear', 'YScale', 'linear');
set(ax1, 'XTick', 0:0.5:6); 

yl_B = ylim(ax1);
default_yticks = get(ax1, 'YTick');          
y_ticks = unique(sort([0, default_yticks]));
set(ax1, 'YTick', y_ticks);


text(ax1, 2.05, yl_B(1) + 0.05*diff(yl_B), '2 periods', 'FontSize',8.5, 'Color',[0.3 0.3 0.3]);
xlabel(ax1, 'window length T (periods)');
ylabel(ax1, 'calculated {\xi_R} coefficient');
title(ax1, 'Convergence on {\xi_R}=15', 'FontWeight','normal');
legend(leg_h, arrayfun(@(p) sprintf('start phase = %.2f', p), start_phases_frac, 'UniformOutput', false), ...
    'Location','northeast', 'FontSize', 8, 'Box','off');
xlim(ax1, [0 4]);

%% --- Right Panel: 2D heatmap (-50% to +50% range) ---------------------
ax2 = nexttile(tl, 2);
[XX, YY] = meshgrid(window_lengths2, start_phases2);

% Plot raw bias bounded directly between -50% and +50%
pcolor(ax2, XX, YY, bias_map);
shading(ax2, 'flat');
colormap(ax2, redbluecmap_local());

% Explicitly set color limit from -50 to 50
clim(ax2, [-50 50]);

cb = colorbar(ax2);
cb.Label.String = 'coefficient bias (%)';
cb.Ticks = -150:10:150; % Set clear increments on colorbar

hold(ax2,'on');
xline(ax2, 2, 'k:', 'LineWidth', 1.3);
xlabel(ax2, 'window length T (periods)');
ylabel(ax2, 'window start phase (% of period)');
title(ax2,  'Start phase and window length bias in  {\xi_R}', 'FontWeight','normal');
box(ax2,'on');

set(findall(gcf, '-property', 'FontSize'), 'FontSize', 18);


%% ---------------- Local helper: diverging red-blue colormap ------------
function cmap = redbluecmap_local()
    n_half = 128;
    top = [linspace(0.10,1,n_half)', linspace(0.20,1,n_half)', linspace(0.55,1,n_half)']; % blue -> white
    bot = [linspace(1,0.70,n_half)', linspace(1,0.10,n_half)', linspace(1,0.10,n_half)']; % white -> red
    cmap = [top; bot];
end