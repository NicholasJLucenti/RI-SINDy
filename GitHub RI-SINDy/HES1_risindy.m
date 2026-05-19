close all; clc;
addpath(genpath('.'));

%% --- DATA ---
dt  = 15 / 301; 
tau = 0.25;
N   = 280;

load("interpHes1Data.mat");
load("interpmRNAData.mat");

t_train = linspace(0, 15, 301)';
x       = interpmRNAData;   % 400 points, t = 0-15 hrs
y       = interpHes1Data;   % 400 points, t = 0-15 hrs

%% --- LIBRARY ---

hill_n    = 9;
hill_k0   = 2.7;
hill_func = @(p) 1 ./ (1 + (p./hill_k0).^hill_n);

polyorder = 2;
y_tau     = get_delayed(t_train, y, tau);
HillDelay = hill_func(y_tau);

Theta = [build_poly_library(x, y, polyorder), HillDelay];
Theta = Theta(1:N, :);

dxdt = smooth_derivative(x, dt, 3, 11);
dydt = smooth_derivative(y, dt, 3, 11);
dXdt = [dxdt(1:N), dydt(1:N)];

%% --- CONSTRAINTS ---
lb{1} = [  0, -inf, -inf,  0,  0];
ub{1} = [  0, -1.5,    0,  0,  0];
lb{2} = [  0,    0,    0, -inf, -inf];
ub{2} = [inf,  inf,  inf,    0,    0];

%% --- AUXILIARY FUNCTION ---
aux_fn = @(XiN, XiN_var, vi, col_scale, target_scale) ...
    aux_hes1(XiN, XiN_var, vi, col_scale, target_scale, ...
             x(1:N), y(1:N), hill_k0, hill_n);

%% --- RUN ---
pinned_col = size(Theta, 2);   % Hill term is last column
[Xi, Xi_var] = risindy(Theta, dXdt, aux_fn, pinned_col, lb, ub, 0.4, 20, 'SpikeSlab');

disp(Xi);

%% --- FORWARD INTEGRATION ---
XiX = Xi(:, 1);  XiY = Xi(:, 2);
x_id = zeros(size(interpmRNAData));  x_id(1) = x(1);
y_id = zeros(size(interpHes1Data));  y_id(1) = y(1);
delay_s = round(tau/dt);

for k = 2:length(interpmRNAData)
    xp = x_id(k-1); yp = y_id(k-1);
    if k <= delay_s + 1
        y_tau_k = interp1(t_train, y, max(t_train(k)-tau, 0), 'linear');
    else
        y_tau_k = y_id(k - delay_s);
    end
    phi     = [poly_library_row(xp, yp, polyorder), hill_func(y_tau_k)];
    x_id(k) = xp + dt * (phi * XiX);
    y_id(k) = yp + dt * (phi * XiY);
end

figure
subplot(1,2,1); hold on;
plot(t_train, x,    'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_train, x_id, 'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(N), 'k:', 'LineWidth', 1.2, 'DisplayName', 'Training end');
xlabel('Time (hrs)'); ylabel('mRNA Concentration');
title('mRNA'); legend('Location','best'); grid on;
xlim([0, 15]);

subplot(1,2,2); hold on;
plot(t_train, y,    'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_train, y_id, 'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(N), 'k:', 'LineWidth', 1.2, 'DisplayName', 'Training end');
xlabel('Time (hrs)'); ylabel('Hes1 Concentration');
title('Hes1 Protein'); legend('Location','best'); grid on;
xlim([0, 15]);

sgtitle('RI-SINDy Identified Trajectories');
