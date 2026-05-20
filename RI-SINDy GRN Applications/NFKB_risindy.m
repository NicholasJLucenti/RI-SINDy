clear; close all; clc;
addpath(genpath('.'));

%% --- PARAMETERS ---
dt      = 0.0375;
N       = 800;
eta     = 0.6;
hill_n  = 3;
hill_k0 = 1.5;

hill_act = @(u, k, n) u.^n ./ (k^n + u.^n);

alpha_z = 8.0;   kxy = 1.2;   kx  = 0.4;
ky_z    = 1.8;   ky  = 0.3;   kz  = 0.8;

%% --- SIMULATE GROUND TRUTH ---
t_span    = 0 : dt : (N + 200)*dt;
signal_fn = @(t) 3.5 * exp(-0.15*t) .* (t > 0);

ode_rhs = @(t, s) [ signal_fn(t) - kxy*s(1)*s(2) - kx*s(1);
                    ky_z*s(3)    - kxy*s(1)*s(2)  - ky*s(2);
                    alpha_z * hill_act(s(1), hill_k0, hill_n) - kz*s(3) ];

[t_all, S] = ode45(ode_rhs, t_span, [0.1; 0.05; 0.02], ...
                   odeset('RelTol',1e-9,'AbsTol',1e-11));
n_all      = length(t_all);
signal_all = signal_fn(t_all);

rng(42);
x_all = S(:,1) + 0.05*std(S(:,1))*randn(n_all,1);
y_all = S(:,2) + 0.05*std(S(:,2))*randn(n_all,1);
z_all = S(:,3) + 0.05*std(S(:,3))*randn(n_all,1);

x = x_all(1:N);   y = y_all(1:N);   z = z_all(1:N);
t_train      = t_all(1:N);
signal_train = signal_all(1:N);

%% --- LIBRARY ---
HillAct  = hill_act(x, hill_k0, hill_n);

Theta_xy = [ones(N,1), signal_train, x, x.^2, x.^3, ...
            y, y.^2, y.^3, x.*y, HillAct];

Theta_z  = [ones(N,1), signal_train, x, x.^2, x.^3, ...
            z, z.^2, z.^3, zeros(N,1), HillAct];

dxdt = smooth_derivative(x, dt, 3, 11);
dydt = smooth_derivative(y, dt, 3, 11);
dzdt = smooth_derivative(z, dt, 3, 11);

%% --- CONSTRAINTS ---
%          1    S     x    x2   x3    v2   v2^2 v2^3  cross
lb{1} = [  0,   0, -inf, -inf, -inf, -inf, -inf, -inf, -inf];
ub{1} = [  0, inf,    0,    0,    0,    0,    0,    0,    0];

lb{2} = [  0,   0,    0,    0,    0, -inf, -inf, -inf, -inf];
ub{2} = [inf,   0,  inf,  inf,  inf,    0,    0,    0,    0];

lb{3} = [  0,   0, -inf, -inf, -inf, -inf, -inf, -inf,    0];
ub{3} = [inf,   0,    0,    0,    0,    0,    0,    0,    0];

%% --- AUXILIARY FUNCTIONS ---
pinned_col = 10;

aux_xy = @(XiN, XiN_var, vi, col_scale, target_scale) ...
    aux_NFkB(XiN, XiN_var, vi, col_scale, target_scale, ...
             x, z, hill_k0, hill_n, 'xy');

aux_z = @(XiN, XiN_var, vi, col_scale, target_scale) ...
    aux_NFkB(XiN, XiN_var, vi, col_scale, target_scale, ...
             x, z, hill_k0, hill_n, 'z');

%% --- RUN ---
[Xi_xy, Xi_var_xy] = risindy(Theta_xy, [dxdt, dydt], aux_xy, pinned_col, ...
                              lb(1:2), ub(1:2), eta, 20, 'SpikeSlab');

[Xi_z, Xi_var_z]   = risindy(Theta_z,  dzdt, aux_z, pinned_col, ...
                              lb(3), ub(3), eta, 20, 'SpikeSlab');

Xi     = [Xi_xy, Xi_z];
Xi_var = [Xi_var_xy, Xi_var_z];
disp(Xi);

XiX = Xi(:,1);   XiY = Xi(:,2);   XiZ = Xi(:,3);

%% --- FORWARD INTEGRATION ---
x_id = zeros(n_all,1);  x_id(1) = x_all(1);
y_id = zeros(n_all,1);  y_id(1) = y_all(1);
z_id = zeros(n_all,1);  z_id(1) = z_all(1);

for k = 2:n_all
    xp = x_id(k-1);  yp = y_id(k-1);  zp = z_id(k-1);
    sp = signal_all(k-1);
    Hr = hill_act(xp, hill_k0, hill_n);

    phi_xy = [1, sp, xp, xp^2, xp^3, yp, yp^2, yp^3, xp*yp, Hr];
    phi_z  = [1, sp, xp, xp^2, xp^3, zp, zp^2, zp^3, 0,     Hr];

    x_id(k) = xp + dt * (phi_xy * XiX);
    y_id(k) = yp + dt * (phi_xy * XiY);
    z_id(k) = zp + dt * (phi_z  * XiZ);
end

%% --- PLOT ---
figure;
subplot(3,1,1); hold on;
plot(t_all, x_all, 'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_all, x_id,  'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(end), 'k:', 'LineWidth', 1.2, 'DisplayName', 'Training end');
ylabel('x (NF-\kappaB)'); legend('Location','best'); grid on; box on;

subplot(3,1,2); hold on;
plot(t_all, y_all, 'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_all, y_id,  'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(end), 'k:', 'LineWidth', 1.2);
ylabel('y (I\kappaB\alpha protein)'); legend('Location','best'); grid on; box on;

subplot(3,1,3); hold on;
plot(t_all, z_all, 'b',   'LineWidth', 1.5, 'DisplayName', 'Data');
plot(t_all, z_id,  'r--', 'LineWidth', 1.5, 'DisplayName', 'RI-SINDy');
xline(t_train(end), 'k:', 'LineWidth', 1.2);
xlabel('Time'); ylabel('z (I\kappaB\alpha mRNA)');
legend('Location','best'); grid on; box on;
sgtitle('RI-SINDy Identified Trajectories — NF-\kappaB Network');

figure('Color','w');
plot3(x_all, y_all, z_all, 'b-',  'LineWidth', 0.8); hold on;
plot3(x_id,  y_id,  z_id,  'r--', 'LineWidth', 1.5);
xlabel('x (NF-\kappaB)'); ylabel('y (I\kappaB\alpha)'); zlabel('z (mRNA)');
title('Phase Portrait — NF-\kappaB Network');
legend('Data', 'RI-SINDy', 'Location', 'best'); grid on;
