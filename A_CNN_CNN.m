%% 清空环境变量
close all
clear all
clc
 
load('noise_data_wangluo_313_2.mat')
load('A_shuru.mat')
% rng(1);%设置随机种子
% 
% % 随机产生训练集和测试集
% 
% P_train = [];
% T_train = [];
% P_test = [];
% T_test = [];
% for i = 1:4
%     temp_input = data_output((i-1)*2703+1:i*2703,:);
%     temp_output = data_output_K((i-1)*2703+1:i*2703,:);
%     n = randperm(2703);
%     % 训练集——20个样本
%     P_train = [P_train temp_input(n(1:2433),:)'];
%     T_train = [T_train temp_output(n(1:2433),:)'];
%     % 测试集——2个样本
%     P_test = [P_test temp_input(n(2434 : 2703),:)'];
%     T_test = [T_test temp_output(n(2434 : 2703),:)'];
% end

 

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


 lgraph = layerGraph();                     % 回归层
tempLayers = imageInputLayer([6 1 1],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],8,"Name","conv","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution2dLayer([2 1],8,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution2dLayer([2 1],8,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],8,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],16,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_3")
    convolution2dLayer([2 1],16,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_4")
    convolution2dLayer([2 1],16,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],16,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    reluLayer("Name","relu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],16,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_6")
    convolution2dLayer([2 1],16,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_7")
    convolution2dLayer([2 1],16,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([2 1],16,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_9")
    convolution2dLayer([2 1],16,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_10")
    convolution2dLayer([2 1],16,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_11")
    fullyConnectedLayer(1,"Name","fc")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","conv");
lgraph = connectLayers(lgraph,"imageinput","conv_2");
lgraph = connectLayers(lgraph,"batchnorm_3","addition/in2");
lgraph = connectLayers(lgraph,"batchnorm_2","addition/in1");
lgraph = connectLayers(lgraph,"relu_2","conv_4");
lgraph = connectLayers(lgraph,"relu_2","conv_7");
lgraph = connectLayers(lgraph,"batchnorm_6","addition_1/in1");
lgraph = connectLayers(lgraph,"batchnorm_7","addition_1/in2");
lgraph = connectLayers(lgraph,"relu_5","conv_8");
lgraph = connectLayers(lgraph,"relu_5","addition_2/in2");
lgraph = connectLayers(lgraph,"batchnorm_10","addition_2/in1");
lgraph = connectLayers(lgraph,"relu_8","conv_11");
lgraph = connectLayers(lgraph,"relu_8","addition_3/in2");
lgraph = connectLayers(lgraph,"batchnorm_13","addition_3/in1");


%%  参数设置
options = trainingOptions('adam', ...      % SGDM 梯度下降算法
    'MiniBatchSize', 256, ...               % 批大小,每次训练样本个数 32
    'MaxEpochs', 500, ...                 % 最大训练次数 1200
    'InitialLearnRate', 1e-2, ...          % 初始学习率为0.01
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.9, ...        % 学习率下降因子
    'LearnRateDropPeriod', 20, ...        % 经过 800 次训练后 学习率为 0.01 * 0.1
    'Shuffle', 'every-epoch', ... 
    'ValidationData',{p_test,t_test},...% 每次训练打乱数据集
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', true);
 
%%  训练模型
net_cnn = trainNetwork(p_train, t_train, lgraph, options);
% net_cnn = trainNetwork(p_train, t_train, lgraph_1, options);




%%  仿真测试
t_sim1 = predict(net_cnn, p_train);
t_sim2 = predict(net_cnn, p_test );
t_sim1 = t_sim1';
t_sim2 = t_sim2';
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
disp(['CNN训练集数据的RMSE为：', num2str(error1)])
disp(['CNN测试集数据的RMSE为：', num2str(error2)])
%%  相关指标计算
% R2

% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['CNN训练集数据的MAE为：', num2str(mae1)])
disp(['CNN测试集数据的MAE为：', num2str(mae2)])

error_BP1 = abs(T_sim1 - T_train);
error_baifenbi1 = abs(T_sim1 - T_train) ./ T_train;
error_BP2 = abs(T_sim2 - T_test);
error_baifenbi2 = abs(T_sim2 - T_test) ./ T_test;
error_baifenbi1 = sort(error_baifenbi1);
error_baifenbi2 = sort(error_baifenbi2);
mdape1 = (error_baifenbi1(4866) + error_baifenbi1(4867))/2*100;
mdape2 = (error_baifenbi2(540) + error_baifenbi2(541))/2*100;
disp(['CNN训练集数据的MdAPE为：', num2str(mdape1)])
disp(['CNN测试集数据的MdAPE为：', num2str(mdape2)])
% MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% disp(['BP训练集数据的MBE为：', num2str(mbe1)])
% disp(['BP测试集数据的MBE为：', num2str(mbe2)])
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['CNN训练集数据的R2为：', num2str(R1)])
disp(['CNN测试集数据的R2为：', num2str(R2)])

%%  查看网络结构
% view(net_grnn)
error_BP = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
disp(['MAXerr：', num2str(max(error_BP))])
disp(['Minerr：', num2str(min(error_BP))])
disp(['MRE：', num2str(mean(100*error_baifenbi))])
% save A_resCNN T_sim1 T_train T_sim2 T_test error_BP

figure
plot(1: N, error_BP, 'b-o', 'LineWidth', 1);
legend('误差绝对值');
xlabel('测试集样本编号');
ylabel('误差绝对值 / K');
% string = {'BP测试集样本误差绝对值'};
% title(string);
xlim([1, N]);
grid;
% %%  数据反归一化
% T_sim1 = mapminmax('reverse', t_sim1, ps_output);
% T_sim2 = mapminmax('reverse', t_sim2, ps_output);
% %%  均方根误差
% error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
% error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
% 
% error_cnn = abs(T_sim2 - T_test);
% error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
% 
% %%  查看网络结构
% 
% 
% 
% 
% figure
% plot(1: N, error_cnn, 'b-o', 'LineWidth', 1);
% legend('误差绝对值');
% xlabel('测试集样本编号');
% ylabel('误差绝对值 / K');
% string = {'CNN测试集样本误差绝对值'};
% title(string);
% xlim([1, N]);
% grid;
% 
% 
% figure
% plot(1: N, error_baifenbi, 'r-*', 'LineWidth', 1);
% legend('误差百分比比值');
% xlabel('测试集样本编号');
% ylabel('误差百分比比值');
% string = {'CNN测试集样本误差百分比比值'};
% title(string);
% xlim([1, N]);
% grid;
% 
% figure
% plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
% legend('真实值', '预测值');
% xlabel('预测样本');
% ylabel('预测结果 / K');
% string = {'CNN训练集预测结果对比'; ['RMSE=' num2str(error1)]};
% title(string);
% xlim([1, M]);
% grid;
% 
% figure
% plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果 / K')
% string = {'CNN测试集预测结果对比'; ['RMSE=' num2str(error2)]};
% title(string)
% xlim([1, N])
% grid
% 
% %%  相关指标计算
% % R2
% R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
% disp(['CNN训练集数据的R2为：', num2str(R1)])
% disp(['CNN测试集数据的R2为：', num2str(R2)])
% % MAE
% mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
% disp(['CNN训练集数据的MAE为：', num2str(mae1)])
% disp(['CNN测试集数据的MAE为：', num2str(mae2)])
% % MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% disp(['CNN训练集数据的MBE为：', num2str(mbe1)])
% disp(['CNN测试集数据的MBE为：', num2str(mbe2)])
% %%  绘制散点图
% sz = 25;
% c = 'b';
% 
% figure
% scatter(T_train, T_sim1, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('CNN训练集真实值');
% ylabel('CNN训练集预测值 / K');
% xlim([min(T_train) max(T_train)])
% ylim([min(T_sim1) max(T_sim1)])
% title('CNN训练集预测值 vs. 训练集真实值')
% 
% figure
% scatter(T_test, T_sim2, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('CNN测试集真实值');
% ylabel('CNN测试集预测值 / K');
% xlim([min(T_test) max(T_test)])
% ylim([min(T_sim2) max(T_sim2)])
% title('CNN测试集预测值 vs. 测试集真实值')

% save CNN_weights net_cnn;
% save resCNN_weights net_cnn;