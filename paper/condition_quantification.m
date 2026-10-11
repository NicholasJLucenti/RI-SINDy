% hill_conditioning_low_degree.m
% (a) Condition-number penalty of adding Hill terms to a low-degree polynomial
%     library, computed via QR decomposition of the Z-SCORE STANDARDIZED
%     library (mean-centered, unit variance per column) -- replaces the
%     unit-L2-norm ("colnorm") normalization used previously. kappa is taken
%     as cond(R) from the economy QR factorization, which equals cond of the
%     standardized matrix itself (Q is orthonormal) but is computed via QR
%     for consistency with the QR-based diagnostic used elsewhere.
% (a2) Ridge-regularized penalty: kappa_lambda = sqrt[(sigma_max^2+lambda)/
%      (sigma_min^2+lambda)] from Theta'Theta + lambda*I, with lambda set
%      per-degree to sigma_min^2 at the worst case (m = max Hill terms) for
%      that degree. The square root puts this on the same scale as cond(A)
%      (a ratio of singular values, not squared singular values / eigenvalues
%      of Theta'Theta) so it is directly comparable to (a) and (a3). This is
%      NOT the same matrix as (a)/(a3) regardless -- see the shrinkage
%      caveat printed below the table.
% (a3) Same penalty on the RAW, unnormalized library, for comparison.
% Verification only; no disk writes. Data: Goodwin limit cycle (n = 12).
clear; clc;
% ---- Data ------------------------------------------------------------------
n = 10; a = 10; b = 1; c = 0.5; K = 1;
f = @(t,s) [a/(1+(s(3)/K)^n) - b*s(1); b*s(1) - c*s(2); c*s(2) - c*s(3)];
[~,S] = ode45(f, 0:0.05:800, [0.1 0.2 0.5], odeset('RelTol',1e-10,'AbsTol',1e-10));
X = S(8001:end,:);
% ---- Hill terms: [variable, threshold quantile, 1 = repression / 0 = activation]
spec = [3 0.5 1; 1 0.5 0; 2 0.5 1; 3 0.7 0; 1 0.3 1; 2 0.7 0];
Hc = zeros(size(X,1), size(spec,1));
for j = 1:size(spec,1)
    x = X(:,spec(j,1)); Kj = qtl(x, spec(j,2));
if spec(j,3), Hc(:,j) = 1./(1 + (x/Kj).^n);
else,         Hc(:,j) = x.^n ./ (Kj^n + x.^n); end
end
% ---- (a) kappa vs number of Hill terms, degrees 1-3, QR ON STANDARDIZED LIB --
%      NOTE: the constant column from poly_lib is dropped before
%      standardizing. Mean-centering a column of ones produces an exact
%      zero column (rank-deficient -> cond = Inf), and the intercept
%      carries no separate information once every column is centered.
dA = 1:3; m = 0:size(spec,1);
kapQR    = zeros(numel(dA), numel(m));
kapRaw   = zeros(numel(dA), numel(m));   % (a3) same thing, no preprocessing at all
kapRidge = zeros(numel(dA), numel(m));   % (a2) ridge, lambda ~= sigma_min^2 per degree
shrink   = zeros(numel(dA), numel(m));   % shrinkage factor, printed as a caveat
lambda_by_degree = zeros(numel(dA), 1);
for i = 1:numel(dA)
    P = poly_lib(X, dA(i));
    Pnc = P(:,2:end);                  % drop constant column for the QR/standardized version

    % lambda for this degree: sigma_min^2 of the raw library at the worst
    % case (m = max), matching the single-lambda choice used previously.
    A_worst = [P Hc(:,1:m(end))];
    sv2_worst = svd(A_worst).^2;
    lambda_by_degree(i) = min(sv2_worst);

for j = 1:numel(m)
        A     = [P Hc(:,1:m(j))];      % raw version keeps the constant column
        Anc   = [Pnc Hc(:,1:m(j))];    % standardized version does not
        kapQR(i,j)  = qr_cond(stdz(Anc));
        kapRaw(i,j) = cond(A);

        sv2 = svd(A).^2;
        s2min_j = min(sv2); s2max_j = max(sv2);
        lam = lambda_by_degree(i);
        % sqrt() converts this from a ratio of eigenvalues of Theta'Theta
        % (squared singular values) to a ratio of singular values, so it
        % is on the same scale as cond(A) used for (a) and (a3).
        kapRidge(i,j) = sqrt((s2max_j + lam) / (s2min_j + lam));
        shrink(i,j)   = s2min_j / (s2min_j + lam);
end
end
% ---- Table ------------------------------------------------------------------
fprintf('\nkappa(Theta), QR on standardized library, vs number of Hill terms m\n%3s', 'd');
for k = m, fprintf('%10s', sprintf('m=%d',k)); end
fprintf('\n');
for i = 1:numel(dA), fprintf('%3d', dA(i)); fprintf('%10.1e', kapQR(i,:)); fprintf('\n'); end
fprintf('\nkappa ratio to polynomial-only, QR/standardized\n');
for i = 1:numel(dA), fprintf('%3d', dA(i)); fprintf('%10.1f', kapQR(i,:)/kapQR(i,1)); fprintf('\n'); end

fprintf('\nkappa_lambda(Theta), ridge (lambda ~= sigma_min^2 per degree, sqrt-scaled), vs number of Hill terms m\n%3s', 'd');
for k = m, fprintf('%10s', sprintf('m=%d',k)); end
fprintf('\n');
for i = 1:numel(dA)
    fprintf('%3d [lam=%.2e]', dA(i), lambda_by_degree(i));
    fprintf('%10.1e', kapRidge(i,:)); fprintf('\n');
end
fprintf('\nkappa ratio to polynomial-only, ridge\n');
for i = 1:numel(dA), fprintf('%3d', dA(i)); fprintf('%10.1f', kapRidge(i,:)/kapRidge(i,1)); fprintf('\n'); end
fprintf('\nRidge shrinkage on the worst direction, sigma_min^2/(sigma_min^2+lambda)\n');
fprintf('(near 0 = that direction is suppressed, not resolved; near 1 = negligible regularization)\n%3s', 'd');
for k = m, fprintf('%10s', sprintf('m=%d',k)); end
fprintf('\n');
for i = 1:numel(dA), fprintf('%3d', dA(i)); fprintf('%10.4f', shrink(i,:)); fprintf('\n'); end

fprintf('\nkappa(Theta), RAW library, vs number of Hill terms m\n%3s', 'd');
for k = m, fprintf('%10s', sprintf('m=%d',k)); end
fprintf('\n');
for i = 1:numel(dA), fprintf('%3d', dA(i)); fprintf('%10.1e', kapRaw(i,:)); fprintf('\n'); end
fprintf('\nkappa ratio to polynomial-only, RAW library\n');
for i = 1:numel(dA), fprintf('%3d', dA(i)); fprintf('%10.1f', kapRaw(i,:)/kapRaw(i,1)); fprintf('\n'); end
% ---- Figure ---------------------------------------------------------------
figure('Color','w','Position',[100 100 1300 340]);
ax1 = subplot(1,3,1);
semilogy(m, kapQR ./ kapQR(:,1), '-o', 'LineWidth', 1.5);
xlabel('Number of Hill terms'); ylabel('\kappa / \kappa_{B}');
legend(arrayfun(@(k) sprintf('d = %d', k), dA, 'UniformOutput', false), 'Location', 'northwest');
title('(a) Condition scaling \kappa / \kappa_{B} Standardized'); grid on;
ax2 = subplot(1,3,2);
semilogy(m, kapRidge ./ kapRidge(:,1), '-o', 'LineWidth', 1.5);
xlabel('Number of Hill terms'); ylabel('\kappa / \kappa_{B}');
legend(arrayfun(@(k) sprintf('d = %d', k), dA, 'UniformOutput', false), 'Location', 'northwest');
title('(a2) Condition scaling \kappa / \kappa_{B} Ridge'); grid on;
ax3 = subplot(1,3,3);
semilogy(m, kapRaw ./ kapRaw(:,1), '-o', 'LineWidth', 1.5);
xlabel('Number of Hill terms'); ylabel('\kappa / \kappa_{B}');
legend(arrayfun(@(k) sprintf('d = %d', k), dA, 'UniformOutput', false), 'Location', 'northwest');
title('(a3) Condition scaling \kappa / \kappa_{B}'); grid on;
linkaxes([ax1 ax2 ax3], 'y');
ylim(ax1, [1e0 1e2]);
yticks(ax1, [1e0 1e1 1e2]);
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 16);
% ---- Local functions --------------------------------------------------------
function P = poly_lib(X, d)
    nv = size(X,2);
    P = ones(size(X,1), 1);
for k = 1:d
        idx = nchoosek(1:nv+k-1, k) - (0:k-1);
for r = 1:size(idx,1)
            P = [P prod(X(:, idx(r,:)), 2)]; %#ok<AGROW>
end
end
end
function A = colnorm(A)
    A = A ./ vecnorm(A);
end
function As = stdz(A)
    % Mean-center and scale to unit variance, column by column.
    mu = mean(A,1); sd = std(A,0,1); sd(sd==0) = 1;
    As = (A - mu) ./ sd;
end
function k = qr_cond(A)
    % Condition number via the R factor of an economy QR decomposition.
    % cond(R) == cond(A) since Q has orthonormal columns; this exists so
    % the computation is explicitly framed as a QR-based diagnostic
    % rather than a direct cond() call on A itself.
    [~, R] = qr(A, 0);
    k = cond(R);
end
function v = qtl(x, q)
    xs = sort(x);
    v = xs(max(1, round(q*numel(xs))));
end

function plot_generic_heatmap(M, labels, ttl)
imagesc(M, [-1 1]);
axis square;
colormap(parula);
cb = colorbar; cb.Label.String = 'correlation';
n = numel(labels);
set(gca, 'XTick', 1:n, 'XTickLabel', labels, 'XTickLabelRotation', 45, ...
    'YTick', 1:n, 'YTickLabel', labels, 'TickLabelInterpreter','tex');
title(ttl, 'Interpreter','tex');
end

function names = poly_lib_names(nv, d, vars)
% Mirrors poly_lib's column-generation order exactly, so labels line
% up with poly_lib(X,d)'s actual columns.
names = {'1'};
for k = 1:d
    idx = nchoosek(1:nv+k-1, k) - (0:k-1);
    for r = 1:size(idx,1)
        parts = vars(idx(r,:));
        names{end+1} = combine_term(parts); %#ok<AGROW>
    end
end
end

function s = combine_term(parts)
u = unique(parts, 'stable');
s = '';
for i = 1:numel(u)
    cnt = sum(strcmp(parts, u{i}));
    if cnt == 1
        s = [s u{i}]; %#ok<AGROW>
    else
        s = [s u{i} '^' num2str(cnt)]; %#ok<AGROW>
    end
end
end

function names = hill_lib_names(spec, vars)
names = cell(1, size(spec,1));
for j = 1:size(spec,1)
    v = vars{spec(j,1)};
    q = spec(j,2);
    if spec(j,3), ttype = 'rep'; else, ttype = 'act'; end
    names{j} = sprintf('H_{%s}(%s,Q%.0f)', ttype, v, 100*q);
end
end