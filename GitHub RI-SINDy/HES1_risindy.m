close all; clc;
addpath(genpath('.'));

%% --- DATA ---
dt  = 0.05;
tau = 0.25;
N   = 290;

load("FH3mRNA_EXTRAP.mat");
load("FH3hes1_EXTRAP.mat");

t_train = (0:dt:15)';
t_all   = (0:dt:35)';

x_all = FH3mRNA_EXTRAP(1:10:end);
y_all = FH3hes1_EXTRAP(1:10:end);
x     = x_all(1:length(t_train));
y     = y_all(1:length(t_train));

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
x_id = zeros(size(x_all));  x_id(1) = x(1);
y_id = zeros(size(y_all));  y_id(1) = y(1);
delay_s = round(tau/dt);

for k = 2:length(t_all)
    xp = x_id(k-1); yp = y_id(k-1);
    if k <= delay_s + 1
        y_tau_k = interp1(t_train, y, max(t_all(k)-tau, 0), 'linear');
    else
        y_tau_k = y_id(k - delay_s);
    end
    phi     = [poly_library_row(xp, yp, polyorder), hill_func(y_tau_k)];
    x_id(k) = xp + dt * (phi * XiX);
    y_id(k) = yp + dt * (phi * XiY);
end

%% --- PLOT ---
figure;
subplot(1,2,1); hold on;
plot(t_all, x_all, 'b', t_all, x_id, 'r--');
xline(t_train(N), 'k:'); xlabel('Time'); ylabel('mRNA'); legend('Data','RI-SINDy'); grid on;
xlim([0,12])

subplot(1,2,2); hold on;
plot(t_all, y_all, 'b', t_all, y_id, 'r--');
xline(t_train(N), 'k:'); xlabel('Time'); ylabel('Hes1'); legend('Data','RI-SINDy'); grid on;
xlim([0,12])