%% 清空环境变量
close all
clear all
clc
 
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

 

%% GRNN创建及仿真测试

M = size(P_train, 2);
N = size(P_test, 2);
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


net_grnn = newgrnn(p_train,t_train, 0.05);
% nftool
net_bp=newff(p_train,t_train,[2,2,2],{'tansig','purelin'},'trainlm');
net_bp=train(net_bp,p_train,t_train);
rbf_spread = 100;                           % 径向基函数的扩展速度
% rbf_spread = mean(sqrt(sumsqr(p_train - mean(p_train,2))));
net_rbf = newrbe(p_train, t_train, rbf_spread);

test_grnn = sim(net_grnn, p_test);
test_bp = sim(net_bp, p_test);
test_rbf = sim(net_rbf,p_test);
train_grnn = sim(net_grnn, p_train);
train_bp = sim(net_bp, p_train);
train_rbf = sim(net_rbf,p_train);


% Y_p = (test_grnn + test_bp + test_rbf);
% T_ppp = mapminmax('reverse', Y_p, ps_output);




X_stack = [train_grnn;train_bp;train_rbf];
X_stack_test = [test_grnn;test_bp;test_rbf];
net_stack = newff(X_stack,t_train, [2,2,2],{'tansig','purelin'},'trainlm'); % 创建stacking元模型
view(net_stack);
net_stack = train(net_stack, X_stack, t_train); % 训练元模型

test_stack = sim(net_stack,X_stack_test); % 元模型对组合特征预测

T_test_grnn = mapminmax('reverse', test_grnn, ps_output);
T_test_bp = mapminmax('reverse', test_bp, ps_output);
T_test_rbf = mapminmax('reverse', test_rbf, ps_output);
T_p = (T_test_rbf + T_test_bp + T_test_grnn) ./3;

T_stack = mapminmax('reverse',test_stack,ps_output);

error_p = abs(T_p - T_test);

error_GRNN = abs(T_test_grnn - T_test);
error_BP= abs(T_test_bp - T_test);
error_RBF = abs(T_test_rbf - T_test);
error_stack = abs(T_stack- T_test);

figure
plot(1: N, error_p, 'b-o', 1: N, error_GRNN,'r',1: N, error_BP,'y',1: N, error_RBF,'g',1: N, error_stack,'k');
error_p_fangcha = sqrt(sum((T_p - T_test).^2) ./ N);
error_stack_fangcha = sqrt(sum((T_stack - T_test).^2) ./ N);
% 
% %%  仿真测试
% t_sim1 = sim(net_grnn, p_train);
% t_sim2 = sim(net_grnn, p_test );
% %%  数据反归一化
% T_sim1 = mapminmax('reverse', t_sim1, ps_output);
% T_sim2 = mapminmax('reverse', t_sim2, ps_output);
% %%  均方根误差
% error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
% error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
% 
% error_GRNN = abs(T_sim2 - T_test);
% error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
% 
% %%  查看网络结构
% view(net_grnn)
% 
% 
% 
% figure
% plot(1: N, error_GRNN, 'b-o', 'LineWidth', 1);
% legend('误差绝对值');
% xlabel('测试集样本编号');
% ylabel('误差绝对值 / K');
% string = {'GRNN测试集样本误差绝对值'};
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
% string = {'GRNN测试集样本误差百分比比值'};
% title(string);
% xlim([1, N]);
% grid;
% 
% figure
% plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
% legend('真实值', '预测值');
% xlabel('预测样本');
% ylabel('预测结果 / K');
% string = {'GRNN训练集预测结果对比'; ['RMSE=' num2str(error1)]};
% title(string);
% xlim([1, M]);
% grid;
% 
% figure
% plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果 / K')
% string = {'GRNN测试集预测结果对比'; ['RMSE=' num2str(error2)]};
% title(string)
% xlim([1, N])
% grid
% 
% %%  相关指标计算
% % R2
% R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
% disp(['GRNN训练集数据的R2为：', num2str(R1)])
% disp(['GRNN测试集数据的R2为：', num2str(R2)])
% % MAE
% mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
% disp(['GRNN训练集数据的MAE为：', num2str(mae1)])
% disp(['GRNN测试集数据的MAE为：', num2str(mae2)])
% % MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% disp(['GRNN训练集数据的MBE为：', num2str(mbe1)])
% disp(['GRNN测试集数据的MBE为：', num2str(mbe2)])
% %%  绘制散点图
% sz = 25;
% c = 'b';
% 
% figure
% scatter(T_train, T_sim1, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('GRNN训练集真实值');
% ylabel('GRNN训练集预测值 / K');
% xlim([min(T_train) max(T_train)])
% ylim([min(T_sim1) max(T_sim1)])
% title('GRNN训练集预测值 vs. 训练集真实值')
% 
% figure
% scatter(T_test, T_sim2, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('GRNN测试集真实值');
% ylabel('GRNN测试集预测值 / K');
% xlim([min(T_test) max(T_test)])
% ylim([min(T_sim2) max(T_sim2)])
% title('GRNN测试集预测值 vs. 测试集真实值')
