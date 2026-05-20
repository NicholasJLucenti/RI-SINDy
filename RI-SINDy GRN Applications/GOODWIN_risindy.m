clear; close all; clc;
addpath(genpath('.'));

%% --- PARAMETERS ---
dt      = 0.05;
N       = 300;
eta     = 0.6;
hill_n  = 3;
hill_k0 = 2;
hill_km = 1.0;

hill_rep = @(z, k, n) k^n ./ (k^n + z.^n);
hill_deg = @(y, km)   y   ./ (km  + y);

alpha = 8.0;   d1 = 0.6;
ks    = 2.0;   Vmax = 5.0;
kp    = 1.5;   kd   = 0.8;

%% --- SIMULATE GROUND TRUTH ---
t_span = 0 : dt : (N + 200)*dt;
ode_rhs = @(t, s) [ alpha*hill_rep(s(3), hill_k0, hill_n) - d1*s(1); ...
                    ks*s(1) - Vmax*hill_deg(s(2), hill_km);            ...
                    kp*s(2) - kd*s(3) ];
[t_all, S] = ode45(ode_rhs, t_span, [1.5; 0.5; 1.0], ...
                   odeset('RelTol',1e-9,'AbsTol',1e-11));
n_all = length(t_all);

rng(42);
x_all = S(:,1) + 0.05*std(S(:,1))*randn(n_all,1);
y_all = S(:,2) + 0.05*std(S(:,2))*randn(n_all,1);
z_all = S(:,3) + 0.05*std(S(:,3))*randn(n_all,1);

x = x_all(1:N);
y = y_all(1:N);
z = z_all(1:N);
t_train = t_all(1:N);

%% --- LIBRARY ---
polyorder  = 3;
HillRep    = hill_rep(z, hill_k0, hill_n);
HillDeg    = hill_deg(y, hill_km);
Theta      = [build_poly_library(x, y, z, polyorder), HillRep, HillDeg];
Theta      = Theta(1:N, :);

dxdt = smooth_derivative(x, dt, 3, 11);
dydt = smooth_derivative(y, dt, 3, 11);
dzdt = smooth_derivative(z, dt, 3, 11);
dXdt = [dxdt(1:N), dydt(1:N), dzdt(1:N)];

%% --- CONSTRAINTS ---
%          1    x   x2   x3    y   y2   y3    z   z2   z3  HillRep HillDeg
lb{1} = [  0, -inf,-inf,-inf,  0,  0,   0,   0,  0,   0,     0,     0  ];
ub{1} = [  0,   -1,   0,   0,  0,  0,   0,   0,  0,   0,   inf,     0  ];

lb{2} = [  0,   0,   0,   0, -inf,-inf,-inf, 0,  0,   0,     0,   -inf ];
ub{2} = [  0, inf, inf, inf,   0,  0,   0,  0,  0,   0,     0,     0  ];

lb{3} = [  0,   0,   0,   0,  0,  0,   0, -inf,-inf,-inf,   0,     0  ];
ub{3} = [  0,   0,   0,   0, inf,inf, inf,   0,  0,   0,   0,     0  ];

%% --- AUXILIARY FUNCTION ---
pinned_cols = [size(Theta,2)-1, size(Theta,2)];   % HillRep, HillDeg are last two columns
aux_fn = @(XiN, XiN_var, vi, col_scale, target_scale) ...
    aux_goodwin(XiN, XiN_var, vi, col_scale, target_scale, ...
                x(1:N), y(1:N), z(1:N), hill_k0, hill_n, hill_km);

%% --- RUN ---
[Xi, Xi_var] = risindy(Theta, dXdt, aux_fn, pinned_cols, lb, ub, eta, 50, 'SpikeSlab');
disp(Xi);

%% --- FORWARD INTEGRATION ---
XiX = Xi(:,1);  XiY = Xi(:,2);  XiZ = Xi(:,3);

x_id = zeros(n_all,1);  x_id(1) = x_all(1);
y_id = zeros(n_all,1);  y_id(1) = y_all(1);
z_id = zeros(n_all,1);  z_id(1) = z_all(1);

for k = 2:n_all
    xp = x_id(k-1);  yp = y_id(k-1);  zp = z_id(k-1);
    phi = [poly_library_row(xp, yp, zp, polyorder), ...
           hill_rep(zp, hill_k0, hill_n), ...
           hill_deg(yp, hill_km)];
    x_id(k) = xp + dt * (phi * XiX);
    y_id(k) = yp + dt * (phi * XiY);
    z_id(k) = zp + dt * (phi * XiZ);
end

%% --- PLOT ---
figure;
subplot(3,1,1); hold on;
plot(t_all, x_all, 'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_all, x_id,  'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(end), 'k:', 'LineWidth', 1.2, 'DisplayName', 'Training end');
ylabel('x (mRNA)'); legend('Location','best'); grid on; box on;

subplot(3,1,2); hold on;
plot(t_all, y_all, 'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_all, y_id,  'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(end), 'k:', 'LineWidth', 1.2);
ylabel('y (Protein)'); legend('Location','best'); grid on; box on;

subplot(3,1,3); hold on;
plot(t_all, z_all, 'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_all, z_id,  'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(end), 'k:', 'LineWidth', 1.2);
xlabel('Time (hrs)'); ylabel('z (Repressor)');
legend('Location','best'); grid on; box on;
sgtitle('RI-SINDy Identified Trajectories — Goodwin Oscillator');

figure('Color','w');
plot3(x_all, y_all, z_all, 'b-',  'LineWidth', 0.8); hold on;
plot3(x_id,  y_id,  z_id,  'r--', 'LineWidth', 1.5);
xlabel('x (mRNA)'); ylabel('y (Protein)'); zlabel('z (Repressor)');
title('Phase Portrait — Goodwin Oscillator'); grid on;
legend('Data', 'RI-SINDy', 'Location', 'best');