%% 清空环境变量
close all
clear all
clc
 
load('data_wangluo_313_2.mat');
P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:1080
    temp_input = data_output((i-1)*10+1:i*10,:);
    temp_output = data_output_K((i-1)*10+1:i*10,:);
    
    % 训练集——20个样本
    P_train = [P_train temp_input(1:9,:)'];
    T_train = [T_train temp_output(1:9,:)'];
    % 测试集——2个样本
    P_test = [P_test temp_input(10,:)'];
    T_test = [T_test temp_output(10,:)'];
end
    P_train = [P_train data_output(10801:10812,:)'];
    T_train = [T_train data_output_K(10801:10812,:)'];


    n = randperm(size(P_train,2));
    P_train= P_train(:,n);
    T_train= T_train(:,n);
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
% % rng(0);
% 


% save shuru P_train  T_train P_test T_test ;
%% GRNN创建及仿真测试

M = size(P_train, 2);
N = size(P_test, 2);
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


net_grnn = newgrnn(p_train,t_train, 0.01);
tic;
t_sim2 = sim(net_grnn, p_test);
time_grnn = toc;
t_sim1 = sim(net_grnn, p_train);
% 
% %%  仿真测试
% t_sim1 = sim(net_grnn, p_train);
% t_sim2 = sim(net_grnn, p_test );
%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
disp(['GRNN训练集数据的RMSE为：', num2str(error1)])
disp(['GRNN测试集数据的RMSE为：', num2str(error2)])
%%  相关指标计算
% R2

% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['GRNN训练集数据的MAE为：', num2str(mae1)])
disp(['GRNN测试集数据的MAE为：', num2str(mae2)])

error_BP1 = abs(T_sim1 - T_train);
error_baifenbi1 = abs(T_sim1 - T_train) ./ T_train;
error_BP2 = abs(T_sim2 - T_test);
error_baifenbi2 = abs(T_sim2 - T_test) ./ T_test;
error_baifenbi1 = sort(error_baifenbi1);
error_baifenbi2 = sort(error_baifenbi2);
mdape1 = (error_baifenbi1(4866) + error_baifenbi1(4867))/2*100;
mdape2 = (error_baifenbi2(540) + error_baifenbi2(541))/2*100;
disp(['GRNN训练集数据的MdAPE为：', num2str(mdape1)])
disp(['GRNN测试集数据的MdAPE为：', num2str(mdape2)])
% MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% disp(['BP训练集数据的MBE为：', num2str(mbe1)])
% disp(['BP测试集数据的MBE为：', num2str(mbe2)])
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['GRNN训练集数据的R2为：', num2str(R1)])
disp(['GRNN测试集数据的R2为：', num2str(R2)])

%%  查看网络结构
% view(net_grnn)
error_BP = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
disp(['MAXerr：', num2str(max(error_BP))])
disp(['Minerr：', num2str(min(error_BP))])
disp(['MRE：', num2str(mean(100*error_baifenbi))])
% save B_GRNN T_sim1 T_train T_sim2 T_test error_BP
figure
plot(1: N, error_BP, 'b-o', 'LineWidth', 1);
legend('误差绝对值');
xlabel('测试集样本编号');
ylabel('误差绝对值 / K');
% string = {'BP测试集样本误差绝对值'};
% title(string);
xlim([1, N]);
grid;
% %%  均方根误差
% error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
% error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
% disp(['GRNN训练集数据的RMSE为：', num2str(error1)])
% disp(['GRNN测试集数据的RMSE为：', num2str(error2)])
% % MAE
% mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
% disp(['BP训练集数据的MAE为：', num2str(mae1)])
% disp(['BP测试集数据的MAE为：', num2str(mae2)])
% 
% error_BP1 = abs(T_sim1 - T_train);
% error_baifenbi1 = abs(T_sim1 - T_train) ./ T_train;
% error_BP2 = abs(T_sim2 - T_test);
% error_baifenbi2 = abs(T_sim2 - T_test) ./ T_test;
% error_baifenbi1 = sort(error_baifenbi1);
% error_baifenbi2 = sort(error_baifenbi2);
% mdape1 = (error_baifenbi1(4866) + error_baifenbi1(4867))/2*100;
% mdape2 = (error_baifenbi2(540) + error_baifenbi2(541))/2*100;
% disp(['BP训练集数据的MdAPE为：', num2str(mdape1)])
% disp(['BP测试集数据的MdAPE为：', num2str(mdape2)])
% % MBE
% % mbe1 = sum(T_sim1 - T_train) ./ M ;
% % mbe2 = sum(T_sim2 - T_test ) ./ N ;
% % disp(['BP训练集数据的MBE为：', num2str(mbe1)])
% % disp(['BP测试集数据的MBE为：', num2str(mbe2)])
% R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
% disp(['BP训练集数据的R2为：', num2str(R1)])
% disp(['BP测试集数据的R2为：', num2str(R2)])
% error_GRNN = abs(T_sim2 - T_test);
% error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
% 
% %%  查看网络结构
% % view(net_grnn)
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
