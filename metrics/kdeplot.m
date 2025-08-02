clc;
clear;
load('pred1.mat');
load('pred57.mat');
load('pred113.mat');
load('pred169.mat');
load('pred225.mat');
load('pred278.mat');
load('actual.mat');
[N, M] = size(pred1);

%figure;

subplot(2,3,1);
hold on;
for m = 1:N
    predt1 = pred1(m, :);
    actual1=actual(1,:);
    predt1_min = min(predt1);
    predt1_max = max(predt1);
    predt1_norm = (predt1 - predt1_min) / (predt1_max - predt1_min);
    
    len1=length(predt1_norm);
    tqr1=prctile(predt1_norm,[25,75]);
    IQR1=tqr1(:,2)-tqr1(:,1);
    optimal_bw1 = 1.06*(min(std(predt1_norm) ,IQR1/1.34))*(len1^(-1/5));
    
    [y1, x1] = ksdensity(predt1_norm, 'Bandwidth', optimal_bw1);
    
    z1 = x1 * (predt1_max - predt1_min) + predt1_min;
    
    plot(z1, y1, 'DisplayName', sprintf('Model %d', m),'LineWidth', 1);
end


xline(actual1, 'k--', 'LineWidth', 2);
xlabel('Offshore wind power(MW)');
ylabel('Kernel density');
legend('Location', 'best');
title('1st');
hold off;

subplot(2,3,2);
hold on;
for m = 1:N
    pred2 = pred57(m, :);
    actual2=actual(57,:);
    pred2_min = min(pred2);
    pred2_max = max(pred2);
    pred2_norm = (pred2 - pred2_min) / (pred2_max - pred2_min);
    len2=length(pred2_norm);
    tqr2=prctile(pred2_norm,[25,75]);
    IQR2=tqr2(:,2)-tqr2(:,1);
    optimal_bw2 = 1.06*(min(std(pred2_norm) ,IQR2/1.34))*(len2^(-1/5));
    
    [y2, x2] = ksdensity(pred2_norm, 'Bandwidth', optimal_bw2);
    
    z2 = x2 * (pred2_max - pred2_min) + pred2_min;
    
    
    plot(z2, y2, 'DisplayName', sprintf('Model %d', m),'LineWidth', 1);
end


xline(actual2, 'k--', 'LineWidth', 2);

xlabel('Offshore wind power(MW)');
ylabel('Kernel density');
legend('Location', 'best');
title('57th');
hold off;

subplot(2,3,3);
hold on;
for m = 1:N
    pred3 = pred113(m, :);
    actual113=actual(113,:);
    pred3_min = min(pred3);
    pred3_max = max(pred3);
    pred3_norm = (pred3 - pred3_min) / (pred3_max - pred3_min);
    
    tqr=prctile(pred3_norm,[25,75]);
    IQR3=tqr(:,2)-tqr(:,1);
    len3=length(pred3_norm);
    optimal_bw3 = 1.06*(min(std(pred3_norm) ,IQR3/1.34))*(len3^(-1/5));
    
    [y3, x3] = ksdensity(pred3_norm, 'Bandwidth', optimal_bw3);
    
    
    z3 = x3 * (pred3_max - pred3_min) + pred3_min;
    
    
    plot(z3, y3, 'DisplayName', sprintf('Model %d', m),'LineWidth', 1);
end


xline(actual113, 'k--', 'LineWidth', 2);
xlabel('Offshore wind power(MW)');
ylabel('Kernel density');
legend('Location', 'best');
title('113rd');
hold off;

subplot(2,3,4);
hold on;
for m = 1:N
    pred4 = pred169(m, :);
    actual169=actual(169,:);
    pred4_min = min(pred4);
    pred4_max = max(pred4);
    pred4_norm = (pred4 - pred4_min) / (pred4_max - pred4_min);
    
    tqr=prctile(pred4_norm,[25,75]);
    IQR4=tqr(:,2)-tqr(:,1);
    len4=length(pred4_norm);
    optimal_bw4 = 1.06*(min(std(pred4_norm) ,IQR4/1.34))*(len4^(-1/5));
    
    [y4, x4] = ksdensity(pred4_norm, 'Bandwidth', optimal_bw4);
    
    
    z4 = x4 * (pred4_max - pred4_min) + pred4_min;
    
    
    plot(z4, y4, 'DisplayName', sprintf('Model %d', m),'LineWidth', 1);
end


xline(actual169, 'k--', 'LineWidth', 2);
xlabel('Offshore wind power(MW)');
ylabel('Kernel density');
legend('Location', 'best');
title('169th');
hold off;

subplot(2,3,5);
hold on;
for m = 1:N
    pred5 = pred225(m, :);
    actual225=actual(225,:);
    pred5_min = min(pred5);
    pred5_max = max(pred5);
    pred5_norm = (pred5 - pred5_min) / (pred5_max - pred5_min);
    
    tqr=prctile(pred5_norm,[25,75]);
    IQR5=tqr(:,2)-tqr(:,1);
    len5=length(pred5_norm);
    optimal_bw5 = 1.06*(min(std(pred5_norm) ,IQR5/1.34))*(len5^(-1/5));
    
    [y5, x5] = ksdensity(pred5_norm, 'Bandwidth', optimal_bw5);
    
    
    z5 = x5 * (pred5_max - pred5_min) + pred5_min;
    
    
    plot(z5, y5, 'DisplayName', sprintf('Model %d', m),'LineWidth', 1);
end


xline(actual225, 'k--', 'LineWidth', 2);
xlabel('Offshore wind power(MW)');
ylabel('Kernel density');
legend('Location', 'best');
title('225th');
hold off;

subplot(2,3,6);
hold on;
for m = 1:N
    pred6 = pred278(m, :);
    actual6=actual(278,:);
    pred6_min = min(pred6);
    pred6_max = max(pred6);
    pred6_norm = (pred6 - pred6_min) / (pred6_max - pred6_min);
    
    tqr=prctile(pred6_norm,[25,75]);
    IQR6=tqr(:,2)-tqr(:,1);
    len6=length(pred6_norm);
    optimal_bw6 = 1.06*(min(std(pred6_norm) ,IQR6/1.34))*(len6^(-1/5));
    
    [y6, x6] = ksdensity(pred6_norm, 'Bandwidth', optimal_bw6);
    
    
    z6 = x6 * (pred6_max - pred6_min) + pred6_min;
    
    
    plot(z6, y6, 'DisplayName', sprintf('Model %d', m),'LineWidth', 1);
end


xline(actual6, 'k--', 'LineWidth', 2);
xlabel('Offshore wind power(MW)');
ylabel('Kernel density');
legend('Location', 'best');
title('278th');
hold off;