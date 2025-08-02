function AIS = average_interval_score(observations, lower_bounds, upper_bounds, alpha)
    % 计算 Average Interval Score (AIS)。
    %
    % 参数:
    % observations : array_like
    %     真实观测值。
    % lower_bounds : array_like
    %     预测区间的左边界。
    % upper_bounds : array_like
    %     预测区间的右边界。
    % alpha : float, optional
    %     置信水平，默认为0.1。
    %
    % 返回:
    % AIS : float
    %     平均区间分数。

    % 默认值
    if nargin < 4
        alpha = 0.1;
    end

    % 计算区间宽度
    interval_width = upper_bounds - lower_bounds;

    % 初始化校准分数
    sj = -2 * alpha * interval_width;

    % 找到观测值小于左边界的情况
    idx_left = observations < lower_bounds;
    sj(idx_left) = sj(idx_left) - 4 * (lower_bounds(idx_left) - observations(idx_left));

    % 找到观测值大于右边界的情况
    idx_right = observations > upper_bounds;
    sj(idx_right) = sj(idx_right) - 4 * (observations(idx_right) - upper_bounds(idx_right));

    % 计算归一化的平均偏差
    AIS = mean(sj);
end