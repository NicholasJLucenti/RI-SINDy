%% =========================================================================
%  Hes1 Phase-Space Field Decomposition (supplementary, not in the paper)
%
%  Visualizes the identified RI-SINDy vector field over the (mRNA,
%  Protein) phase plane, split into three panels:
%    1. Total field         -- the full identified dx/dt, dy/dt
%    2. Polynomial (drain) field only -- contribution from the free
%       (regressed) library columns alone, with the Hill/regulatory
%       column's contribution zeroed out
%    3. Regulatory field only -- contribution from the pinned Hill
%       column alone, with every polynomial column zeroed out
%
%  This is the kind of figure that's genuinely useful for understanding
%  WHERE in phase space the drift-balance term is doing the work vs.
%  where the ordinary drain terms dominate, but it takes more space to
%  explain properly than the paper has room for -- kept here as a
%  repo-only supplementary figure instead.
%
%  APPROXIMATION NOTE: Hes1's regulatory term is a function of the
%  DELAYED protein level (y_tau), not the instantaneous y at a given
%  grid point. Off the actual trajectory, "the value tau hours ago at
%  this point" isn't well-defined -- there's no history to look back
%  on at an arbitrary grid location. This figure uses the INSTANTANEOUS
%  y at each grid point as a stand-in for y_tau (a quasi-steady-state
%  approximation), exactly as the original exploratory version of this
%  figure did. It's a reasonable approximation for visualizing the
%  field's general shape, but the quiver arrows are not a literal
%  reproduction of the delayed dynamics -- don't read exact magnitudes
%  off this plot as if delay weren't a factor.
%
%  Requires: nothing beyond src/ and the CSVs in "Hes1 Data".
% =========================================================================

clear; close all; clc;
thisDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(thisDir, '..', '..', 'src')));

%% --- HYPERPARAMETERS AND DATA -------------------------------------------
polyorder = 2;
hill_n    = 9;
hill_k0   = 2.7;
hill_func = @(p,k,n) 1 ./ (1 + (p./k).^n);

dataDir = fullfile(thisDir, '..', '..', 'Hes1 Data');
x_data_all = readmatrix(fullfile(dataDir, 'interpmRNAData.csv'));
y_data_all = readmatrix(fullfile(dataDir, 'interpHes1Data.csv'));

% Already-identified RI-SINDy coefficients (same values used throughout
% paper/hes1/ and paper/comparisons/hes1_comparisons.m) -- this script
% only visualizes them, it does not re-fit anything.
% rows: 1, M, M^2, P, P^2, HillDelay  |  columns: mRNA, Protein
Xi = [   0        ,   0       ; ...   % 1
        -1.90514   ,   0       ; ...   % M
         0         ,   0.64064 ; ...   % M^2
         0         ,  -1.01964 ; ...   % P
         0         ,   0       ; ...   % P^2
        11.76276    ,   0       ];      % HillDelay

free_cols  = 1:5;   % 1, M, M^2, P, P^2
pinned_col = 6;      % HillDelay

%% --- GRID -----------------------------------------------------------------
grid_res = 20;
x_range = linspace(min(x_data_all), max(x_data_all), grid_res);
y_range = linspace(min(y_data_all), max(y_data_all), grid_res);
[Xg, Yg] = meshgrid(x_range, y_range);

field_types = {'Total Identified Field', 'Polynomial Field (Drain)', 'Regulatory Field (Drift-Balance)'};

for f = 1:3
    figure('Color', 'w', 'Name', field_types{f});
    hold on; grid on;

    % Reference trajectory, same on every panel.
    plot(x_data_all, y_data_all, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Identified trajectory');

    U = zeros(size(Xg));
    V = zeros(size(Yg));

    for i = 1:numel(Xg)
        xp = Xg(i); yp = Yg(i);

        % See APPROXIMATION NOTE above: yp stands in for the delayed
        % y_tau here, since there's no trajectory history off-grid.
        H_inst    = hill_func(yp, hill_k0, hill_n);
        poly_part = poly_library_row(xp, yp, polyorder);
        phi_total = [poly_part, H_inst];

        switch f
            case 1   % Total field
                U(i) = phi_total * Xi(:,1);
                V(i) = phi_total * Xi(:,2);
            case 2   % Polynomial (drain) contribution only
                U(i) = poly_part * Xi(free_cols,1);
                V(i) = poly_part * Xi(free_cols,2);
            case 3   % Regulatory (drift-balance) contribution only
                U(i) = H_inst * Xi(pinned_col,1);
                V(i) = H_inst * Xi(pinned_col,2);
        end
    end

    % Normalize arrow length so direction, not magnitude, is what's
    % visually compared across panels -- the three fields have very
    % different natural magnitudes (the total field is a sum of the
    % other two), so an unnormalized quiver would make panels 2 and 3
    % look artificially small next to panel 1.
    L = sqrt(U.^2 + V.^2);
    L(L == 0) = 1;
    quiver(Xg, Yg, U./L, V./L, 0.5, 'Color', [0.55 0.55 0.55], ...
           'AutoScale', 'off', 'DisplayName', field_types{f});

    plot(x_data_all(1), y_data_all(1), 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Start');

    xlabel('Hes1 mRNA'); ylabel('Hes1 Protein');
    title(field_types{f});
    axis tight; box on;
    legend('Location', 'northeast');
end