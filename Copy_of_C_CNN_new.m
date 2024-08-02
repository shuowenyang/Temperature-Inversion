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


layers = [
    % 卷积层1
    imageInputLayer([size(p_train, 1),1 1])

    convolution2dLayer([3 3],64,'Padding','same')
    reluLayer()
    convolution2dLayer([3 3],64,'Padding','same')
    reluLayer()  
    maxPooling2dLayer([1 1],'Stride',1)

    % 卷积层2
    convolution2dLayer([3 3],128,'Padding','same') 
    reluLayer()
    convolution2dLayer([3 3],128,'Padding','same')
    reluLayer()
    maxPooling2dLayer([1 1],'Stride',1) 

    % 卷积层3 
    convolution2dLayer([3 3],256,'Padding','same')
    reluLayer()
    convolution2dLayer([3 3],256,'Padding','same') 
    reluLayer()
    convolution2dLayer([3 3],256,'Padding','same')
    reluLayer()
    maxPooling2dLayer([1 1],'Stride',1)

    % 卷积层4
    convolution2dLayer([3 3],512,'Padding','same')
    reluLayer()
    convolution2dLayer([3 3],512,'Padding','same')
    reluLayer()
    convolution2dLayer([3 3],512,'Padding','same') 
    reluLayer()  
    maxPooling2dLayer([1 1],'Stride',1)

    % 卷积层5
    convolution2dLayer([3 3],512,'Padding','same')
    reluLayer() 
    convolution2dLayer([3 3],512,'Padding','same')
    reluLayer()
    convolution2dLayer([3 3],512,'Padding','same')
    reluLayer()
    maxPooling2dLayer([1 1],'Stride',1)

    fullyConnectedLayer(1)
    regressionLayer]; % 回归层
% net = replaceLayer(net, 'ClassificationLayer_fc1000', newLayers(3));
% 显示修改后的网络结构
% analyzeNetwork(net);



% analyzeNetwork(net5);

%% 5.指定训练选项
   options = trainingOptions('adam', ...%求解器，''（默认） | 'rmsprop' | 'adam'
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',Inf, ...  %梯度极限
    'MaxEpochs',500, ...%最大迭代次数
    'InitialLearnRate', 0.01, ...%初始化学习速率
    'ValidationFrequency',10, ...%验证频率，即每间隔多少次迭代进行一次验证
    'MiniBatchSize',64, ...
    'LearnRateSchedule','piecewise', ...%是否在一定迭代次数后学习速率下降
    'LearnRateDropFactor',0.9, ...%学习速率下降因子
    'LearnRateDropPeriod',10, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'ValidationData',{p_test,t_test},...
    'Verbose',true, ...
    'Plots','training-progress');%显示训练过程

% 训练模型
net_cnn = trainNetwork(p_train, t_train, layers, options);
% net_cnn = trainNetwork(p_train, t_train, lgraph, options);




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
