
 
%% 以下程序为案例扩展里的GRNN和BP比较 需要load chapter8.1的相关数据
close all
clear all
load data_fangzhen.mat;


% 随机产生训练集和测试集

P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:4
    temp_input = data_output((i-1)*310+1:i*310,:);
    temp_output = data_output_K((i-1)*310+1:i*310,:);
    n = randperm(310);
    % 训练集——20个样本
    P_train = [P_train temp_input(n(1:248),:)'];
    T_train = [T_train temp_output(n(1:248),:)'];
    % 测试集——2个样本
    P_test = [P_test temp_input(n(249 : 310),:)'];
    T_test = [T_test temp_output(n(249 : 310),:)'];
end

M = size(P_train, 2);
N = size(P_test, 2);
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

net_bp=newff(p_train,t_train,[2,2,2],{'tansig','purelin'},'trainlm');
% 训练网络%设置训练参数

net.trainParam.lr=0.01;               %学习率
net.trainparam.show=40;                %每训练400次展示一次结果
net.trainParam.goal=0.01;             %目标误差
net.trainparam.epochs=2000;            %最大训练次数
net.trainParam.showWindow=1;            %BP训练窗口
% net.divideParam.trainRatio=0.8;         % 用于训练的数据比例
% net.divideParam.valRatio=0.2;           % 用于验证过拟合的数据比例
% net.divideParam.testRatio=0;            % 注意要关掉测试数据占比

%调用TRAINLM算法训练BP网络
net_bp=train(net_bp,p_train,t_train);
%%  仿真测试
t_sim1 = sim(net_bp, p_train);
t_sim2 = sim(net_bp, p_test );
%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
% error00 = sum((T_sim1 - T_train).^2) ./ M
% mse1 = mse(T_sim1, T_train)
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

error_BP = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;

%%  查看网络结构
% view(net_grnn)

figure
plot(1: N, error_BP, 'b-o', 'LineWidth', 1);
legend('误差绝对值');
xlabel('测试集样本编号');
ylabel('误差绝对值 / K');
string = {'BP测试集样本误差绝对值'};
title(string);
xlim([1, N]);
grid;


figure
plot(1: N, error_baifenbi, 'r-*', 'LineWidth', 1);
legend('误差百分比比值');
xlabel('测试集样本编号');
ylabel('误差百分比比值');
string = {'BP测试集样本误差百分比比值'};
title(string);
xlim([1, N]);
grid;

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值');
xlabel('预测样本');
ylabel('预测结果 / K');
string = {'BP训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string);
xlim([1, M]);
grid;

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果 / K')
string = {'BP测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['BP训练集数据的R2为：', num2str(R1)])
disp(['BP测试集数据的R2为：', num2str(R2)])
% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['BP训练集数据的MAE为：', num2str(mae1)])
disp(['BP测试集数据的MAE为：', num2str(mae2)])
% MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
disp(['BP训练集数据的MBE为：', num2str(mbe1)])
disp(['BP测试集数据的MBE为：', num2str(mbe2)])
%%  绘制散点图
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('BP训练集真实值');
ylabel('BP训练集预测值 / K');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('BP训练集预测值 vs. 训练集真实值')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('BP测试集真实值');
ylabel('BP测试集预测值 / K');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('BP测试集预测值 vs. 测试集真实值')