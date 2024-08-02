%% 清空环境变量
close all
clear all
clc
 
load('data_wangluo_313_2.mat');


% 随机产生训练集和测试集

P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:4
    temp_input = data_output((i-1)*2703+1:i*2703,:);
    temp_output = data_output_K((i-1)*2703+1:i*2703,:);
    n = randperm(2703);
    % 训练集——20个样本
    P_train = [P_train temp_input(n(1:2000),:)'];
    T_train = [T_train temp_output(n(1:2000),:)'];
    % 测试集——2个样本
    P_test = [P_test temp_input(n(2001 : 2703),:)'];
    T_test = [T_test temp_output(n(2001 : 2703),:)'];
end

 

%% GRNN创建及仿真测试

M = size(P_train, 2);
N = size(P_test, 2);
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


p_train =  double(reshape(p_train, 6, 1, 1, M));
p_test  =  double(reshape(p_test , 6, 1, 1, N));
t_train =  double(t_train)';
t_test  =  double(t_test )';


%%  构造网络结构
lgraph = layerGraph();
tempLayers = imageInputLayer([6 1 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    fullyConnectedLayer(20,"Name","fc_2")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(20,"Name","fc_3")
    batchNormalizationLayer("Name","batchnorm_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_1")
    batchNormalizationLayer("Name","batchnorm_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_4")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(20,"Name","fc_5")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(20,"Name","fc_6")
    batchNormalizationLayer("Name","batchnorm_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_8")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_5")
    fullyConnectedLayer(20,"Name","fc_9")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_6")
    fullyConnectedLayer(20,"Name","fc_10")
    batchNormalizationLayer("Name","batchnorm_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_11")
    batchNormalizationLayer("Name","batchnorm_10")
    reluLayer("Name","relu_8")
    fullyConnectedLayer(20,"Name","fc_12")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_9")
    fullyConnectedLayer(20,"Name","fc_13")
    batchNormalizationLayer("Name","batchnorm_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_14")
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_12")
    fullyConnectedLayer(20,"Name","fc_15")
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_13")
    fullyConnectedLayer(20,"Name","fc_16")
    batchNormalizationLayer("Name","batchnorm_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_17")
    batchNormalizationLayer("Name","batchnorm_16")
    reluLayer("Name","relu_15")
    fullyConnectedLayer(20,"Name","fc_18")
    batchNormalizationLayer("Name","batchnorm_17")
    reluLayer("Name","relu_16")
    fullyConnectedLayer(20,"Name","fc_19")
    batchNormalizationLayer("Name","batchnorm_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5")
    reluLayer("Name","relu_23")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_20")
    batchNormalizationLayer("Name","batchnorm_19")
    reluLayer("Name","relu_17")
    fullyConnectedLayer(20,"Name","fc_21")
    batchNormalizationLayer("Name","batchnorm_20")
    reluLayer("Name","relu_18")
    fullyConnectedLayer(20,"Name","fc_22")
    batchNormalizationLayer("Name","batchnorm_21")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_6")
    reluLayer("Name","relu_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_23")
    batchNormalizationLayer("Name","batchnorm_22")
    reluLayer("Name","relu_20")
    fullyConnectedLayer(20,"Name","fc_24")
    batchNormalizationLayer("Name","batchnorm_23")
    reluLayer("Name","relu_21")
    fullyConnectedLayer(20,"Name","fc_25")
    batchNormalizationLayer("Name","batchnorm_24")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_7")
    reluLayer("Name","relu_22")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    fullyConnectedLayer(20,"Name","fc_26")
    batchNormalizationLayer("Name","batchnorm_25")
    reluLayer("Name","relu_24")
    fullyConnectedLayer(20,"Name","fc_27")
    batchNormalizationLayer("Name","batchnorm_26")
    reluLayer("Name","relu_25")
    fullyConnectedLayer(20,"Name","fc_28")
    batchNormalizationLayer("Name","batchnorm_27")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_8")
    reluLayer("Name","relu_26")
    fullyConnectedLayer(1,"Name","fc_7")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","fc");
lgraph = connectLayers(lgraph,"imageinput","fc_1");
lgraph = connectLayers(lgraph,"batchnorm_1","addition_1/in1");
lgraph = connectLayers(lgraph,"batchnorm_3","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_4","fc_4");
lgraph = connectLayers(lgraph,"relu_4","addition/in2");
lgraph = connectLayers(lgraph,"batchnorm_6","addition/in1");
lgraph = connectLayers(lgraph,"relu_11","fc_8");
lgraph = connectLayers(lgraph,"relu_11","addition_2/in2");
lgraph = connectLayers(lgraph,"batchnorm_9","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_7","fc_11");
lgraph = connectLayers(lgraph,"relu_7","addition_3/in2");
lgraph = connectLayers(lgraph,"batchnorm_12","addition_3/in1");
lgraph = connectLayers(lgraph,"relu_10","fc_14");
lgraph = connectLayers(lgraph,"relu_10","addition_4/in2");
lgraph = connectLayers(lgraph,"batchnorm_15","addition_4/in1");
lgraph = connectLayers(lgraph,"relu_14","fc_17");
lgraph = connectLayers(lgraph,"relu_14","addition_5/in2");
lgraph = connectLayers(lgraph,"batchnorm_18","addition_5/in1");
lgraph = connectLayers(lgraph,"relu_23","fc_20");
lgraph = connectLayers(lgraph,"relu_23","addition_6/in2");
lgraph = connectLayers(lgraph,"batchnorm_21","addition_6/in1");
lgraph = connectLayers(lgraph,"relu_19","fc_23");
lgraph = connectLayers(lgraph,"relu_19","addition_7/in2");
lgraph = connectLayers(lgraph,"batchnorm_24","addition_7/in1");
lgraph = connectLayers(lgraph,"relu_22","fc_26");
lgraph = connectLayers(lgraph,"relu_22","addition_8/in2");
lgraph = connectLayers(lgraph,"batchnorm_27","addition_8/in1");
% 回归层
 % analyzeNetwork(lgraph);
%%  参数设置
options = trainingOptions('adam', ...      % SGDM 梯度下降算法
    'MiniBatchSize', 256, ...               % 批大小,每次训练样本个数 32
    'MaxEpochs', 300, ...                 % 最大训练次数 1200
    'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.9, ...        % 学习率下降因子
    'LearnRateDropPeriod', 10, ...        % 经过 800 次训练后 学习率为 0.01 * 0.1
    'Shuffle', 'every-epoch', ... 
    'ValidationData',{p_test,t_test},...% 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', true);
 
%%  训练模型
% net_cnn = trainNetwork(p_train, t_train, layers, options);
net_cnn = trainNetwork(p_train, t_train, lgraph, options);





%%  仿真测试
t_sim1 = predict(net_cnn, p_train);
t_sim2 = predict(net_cnn, p_test );
t_sim1 = t_sim1';
t_sim2 = t_sim2';
%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

error_cnn = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;

%%  查看网络结构




figure
plot(1: N, error_cnn, 'b-o', 'LineWidth', 1);
legend('误差绝对值');
xlabel('测试集样本编号');
ylabel('误差绝对值 / K');
string = {'CNN测试集样本误差绝对值'};
title(string);
xlim([1, N]);
grid;


figure
plot(1: N, error_baifenbi, 'r-*', 'LineWidth', 1);
legend('误差百分比比值');
xlabel('测试集样本编号');
ylabel('误差百分比比值');
string = {'CNN测试集样本误差百分比比值'};
title(string);
xlim([1, N]);
grid;

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值');
xlabel('预测样本');
ylabel('预测结果 / K');
string = {'CNN训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string);
xlim([1, M]);
grid;

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果 / K')
string = {'CNN测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['CNN训练集数据的R2为：', num2str(R1)])
disp(['CNN测试集数据的R2为：', num2str(R2)])
% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['CNN训练集数据的MAE为：', num2str(mae1)])
disp(['CNN测试集数据的MAE为：', num2str(mae2)])
% MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
disp(['CNN训练集数据的MBE为：', num2str(mbe1)])
disp(['CNN测试集数据的MBE为：', num2str(mbe2)])
%%  绘制散点图
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('CNN训练集真实值');
ylabel('CNN训练集预测值 / K');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('CNN训练集预测值 vs. 训练集真实值')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('CNN测试集真实值');
ylabel('CNN测试集预测值 / K');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('CNN测试集预测值 vs. 测试集真实值')

% save CNN_weights net_cnn;
