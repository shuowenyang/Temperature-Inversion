%% 加载数据
numSamples = 1000;
inputSize = 6;
outputSize = 1;

X = randn([inputSize, numSamples]);
y = randn([outputSize, numSamples]);

%% 数据预处理 
[X_norm, X_mu, X_sigma] = zscore(X); % 标准化输入数据

%% 创建装袋集成

bag_bp = fitrensemble(X_norm',y,"Method","Bag");
bag = fitrensemble(X_norm',y,"Method","LSBoost");
bag_rbf = bagging(X_norm',y',100,'RobustFitRBF'); % RBF装袋集成 
bag_grnn = bagging(X_norm',y',100,'FitGRNN'); % GRNN装袋集成

RegressionEnsemble()
%% 装袋集成预测
y_pred_bp = bag_bp.predict(X_norm'); % BP装袋集成预测
y_pred_rbf = bag_rbf.predict(X_norm'); % RBF装袋集成预测
y_pred_grnn = bag_grnn.predict(X_norm'); % GRNN装袋集成预测

%% 加权平均预测结果  
w_bp = 0.3; w_rbf = 0.4; w_grnn = 0.3; % 设置权重
y_pred = w_bp*y_pred_bp + w_rbf*y_pred_rbf + w_grnn*y_pred_grnn; % 加权平均