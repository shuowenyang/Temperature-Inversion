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


net_grnn = newgrnn(p_train,t_train, 0.005);
% nftool

net_bp=newff(p_train,t_train,[4,4,4,4,4],{'tansig','purelin'},'trainlm');
net_bp=train(net_bp,p_train,t_train);


% rbf_spread = 100;                           % 径向基函数的扩展速度
% rbf_spread = mean(sqrt(sumsqr(p_train - mean(p_train,2))));
% net_rbf = fitrsvm(p_train', t_train);
trees = 100;                                      % 决策树数目
leaf  = 5;                                        % 最小叶子数
OOBPrediction = 'on';                             % 打开误差图
OOBPredictorImportance = 'on';                    % 计算特征重要性
Method = 'regression';                            % 分类还是回归
net_f = TreeBagger(trees, p_train', t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net_f.OOBPermutedPredictorDeltaError;  % 重要性

% net_rbf = newrb(p_train, t_train, 6*10^-4,rbf_spread);



test_grnn = sim(net_grnn, p_test);
test_bp = sim(net_bp, p_test);

test_f = predict(net_f,p_test');
train_grnn = sim(net_grnn, p_train);
train_bp = sim(net_bp, p_train);

train_f = predict(net_f,p_train');


Y_p = (test_grnn + test_bp  + test_f)./3;
T_ppp = mapminmax('reverse', Y_p, ps_output);




X_stack = [train_grnn;train_bp;train_f'];
X_stack_test = [test_grnn;test_bp;test_f'];
net_stack = newff(X_stack,t_train, [4,4],{'tansig','purelin'},'trainlm'); % 创建stacking元模型
% view(net_stack);
net_stack = train(net_stack, X_stack, t_train); % 训练元模型



T_test_grnn = mapminmax('reverse', test_grnn, ps_output);
T_test_bp = mapminmax('reverse', test_bp, ps_output);

T_test_f = mapminmax('reverse', test_f', ps_output);
T_p = (T_test_f + T_test_bp + T_test_grnn) ./3;


t_sim1 = sim(net_stack,X_stack); % 元模型对组合特征预测
t_sim2 = sim(net_stack,X_stack_test); % 元模型对组合特征预测

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

error_stack = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;

%%  查看网络结构
% view(net_grnn)



figure
plot(1: N, error_stack, 'b-o', 'LineWidth', 1);
legend('误差绝对值');
xlabel('测试集样本编号');
ylabel('误差绝对值 / K');
string = {'stack测试集样本误差绝对值'};
title(string);
xlim([1, N]);
grid;


figure
plot(1: N, error_baifenbi, 'r-*', 'LineWidth', 1);
legend('误差百分比比值');
xlabel('测试集样本编号');
ylabel('误差百分比比值');
string = {'stack测试集样本误差百分比比值'};
title(string);
xlim([1, N]);
grid;

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
legend('真实值', '预测值');
xlabel('预测样本');
ylabel('预测结果 / K');
string = {'stack训练集预测结果对比'; ['RMSE=' num2str(error1)]};
title(string);
xlim([1, M]);
grid;

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果 / K')
string = {'stack测试集预测结果对比'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  相关指标计算
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['stack训练集数据的R2为：', num2str(R1)])
disp(['stack测试集数据的R2为：', num2str(R2)])
% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['stack训练集数据的MAE为：', num2str(mae1)])
disp(['stack测试集数据的MAE为：', num2str(mae2)])
% MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
disp(['stack训练集数据的MBE为：', num2str(mbe1)])
disp(['stack测试集数据的MBE为：', num2str(mbe2)])
%%  绘制散点图
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('stack训练集真实值');
ylabel('stack训练集预测值 / K');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('stack训练集预测值 vs. 训练集真实值')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('stack测试集真实值');
ylabel('stack测试集预测值 / K');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('stack测试集预测值 vs. 测试集真实值')