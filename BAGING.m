%% 加载数据
clear;
load data_fangzhen.mat;
X = data_output;
y = data_output_K;
%% 数据预处理
[X_norm, X_mu, X_sigma] = zscore(X); % 标准化输入数据

%% 设置参数
N = 1; % 集成数量
bag_frac = 0.7; % Bagging采样比例
X_norm = X_norm';
y = y';
%% Bagging集成BP网络
y_pred_bp = zeros(size(y)); % 初始化BP集成预测
for i = 1:N
    idx = randsample(length(y), floor(length(y)*bag_frac), true); % Bagging采样
    net_bp = newff(X_norm(:,idx), y(:,idx), [10]);   % 创建BP网络
    % view(net_bp);
    net_bp = train(net_bp, X_norm(:,idx), y(:,idx));% 训练BP网络
    y_pred_bp = y_pred_bp + sim(net_bp, X_norm)/N; % 集成预测加权平均
end

%% Bagging集成RBF网络
spread = 1;
y_pred_rbf = zeros(size(y)); % 初始化RBF集成预测  
for i = 1:N
    idx = randsample(length(y), floor(length(y)*bag_frac), true);
    net_rbf = newrb(X_norm(:,idx), y(:,idx), 0, spread); % 创建RBF网络
    y_pred_rbf = y_pred_rbf + sim(net_rbf, X_norm)/N; % 集成预测加权平均
end
        
%% Bagging集成GRNN网络
spread = mean(sqrt(sumsqr(X_norm - mean(X_norm,2))));
y_pred_grnn = zeros(size(y)); % 初始化GRNN集成预测
for i = 1:N  
    idx = randsample(length(y), floor(length(y)*bag_frac), true);
    net_grnn = newgrnn(X_norm(:,idx), y(:,idx), spread); % 创建GRNN网络
    y_pred_grnn = y_pred_grnn + sim(net_grnn, X_norm)/N; % 集成预测加权平均  
end

%% 加权平均最终集成预测
w_bp = 0.3; w_rbf = 0.4; w_grnn = 0.3; % 设置权重
y_pred = w_bp*y_pred_bp + w_rbf*y_pred_rbf + w_grnn*y_pred_grnn; % 加权平均

error = abs(y_pred - y);