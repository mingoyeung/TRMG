% ---------- 1. 加载数据 ----------
data = load('vmdfujian.mat');%TenneT:tvfemd/wt/vmd/wmd; Fujian: tvfemdfujian/wtfujian/vmdfujian/wmdfujian
X    = data.vmd_fujian;
%TenneT: tvfemdresults, wt_results,vmd_decomposition_results_1,emdresults
%Fujian: tvfemd_fujian, wt_fujian, vmd_fujian, emdf
[n_samples, n_cols] = size(X);
fprintf('数据规模：%d 行 × %d 列\n', n_samples, n_cols);

% ---------- 2. R2CMSE 参数（与原示例一致） ----------
tau = 2;      % 尺度数（计算 1~tau 共 tau 个尺度）
r   = 0.15;   % 容忍度（相对于信号标准差的比例）
m   = 2;      % 模板长度

% ---------- 3. 逐列计算 R2CMSE ----------
% r2cmse_value 返回长度为 tau 的行向量（每个尺度对应一个熵值）
% 这里把所有列的结果堆成矩阵：行=尺度, 列=IMF编号
results = zeros(tau, n_cols);

for col = 1 : n_cols
    signal = X(:, col)';          % 取第 col 列，转成行向量（库要求 1×N）
    results(:, col) = r2cmse_value(signal, r, m, tau)';
    fprintf('列 %2d / %d 完成\n', col, n_cols);
end

% ---------- 4. 显示结果 ----------
fprintf('\n=== R2CMSE 结果（行=尺度 1~%d，列=第1~%d列信号）===\n', tau, n_cols);
disp(results);

