%% 清空环境变量
close all
clear all
clc
 
% load('noise_data_wangluo_313_2.mat')
load('A_shuru.mat')
% P_train = [];
% T_train = [];
% P_test = [];
% T_test = [];
% for i = 1:1080
%     temp_input = data_output((i-1)*10+1:i*10,:);
%     temp_output = data_output_K((i-1)*10+1:i*10,:);
% 
%     % 训练集——20个样本
%     P_train = [P_train temp_input(1:9,:)'];
%     T_train = [T_train temp_output(1:9,:)'];
%     % 测试集——2个样本
%     P_test = [P_test temp_input(10,:)'];
%     T_test = [T_test temp_output(10,:)'];
% end
%     P_train = [P_train data_output(10801:10812,:)'];
%     T_train = [T_train data_output_K(10801:10812,:)'];
% 

%% GRNN创建及仿真测试

M = size(P_train, 2);
N = size(P_test, 2);
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);
[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);


net_grnn = newgrnn(p_train,t_train, 0.015);
% nftool

net_bp=newff(p_train,t_train,[6,6,6],{'tansig','purelin'},'trainlm');
net_bp.trainParam.epochs = 1000; % 训练轮次设置为100
% net_bp.trainParam.miniBatchSize = 100;
net_bp.trainParam.lr = 0.001; % 初始学习率设置为0.001
net_bp=train(net_bp,p_train,t_train);




% rbf_spread = 100;                           % 径向基函数的扩展速度
% rbf_spread = mean(sqrt(sumsqr(p_train - mean(p_train,2))));
% net_rbf = fitrsvm(p_train', t_train);
rbf_spread = 1;      
% rbf_spread = mean(sqrt(sumsqr(p_train - mean(p_train,2))));
net_rbf = newrb(p_train, t_train, 1*10^-4,rbf_spread);
% net_rbf = newrb(p_train, t_train, 6*10^-4,rbf_spread);
load('A_resCNN_weights.mat')
tic;
test_rbf = sim(net_rbf,p_test);
t1 = toc;
train_rbf = sim(net_rbf,p_train);
tic;
test_grnn = sim(net_grnn, p_test);
t2 = toc;
tic;
test_bp = sim(net_bp, p_test);
t3 = toc;

% test_f = predict(net_f,p_test');
train_grnn = sim(net_grnn, p_train);
train_bp = sim(net_bp, p_train);

% train_f = predict(net_f,p_train');

p_train_cnn =  double(reshape(p_train, 6, 1, 1, M));
p_test_cnn  =  double(reshape(p_test , 6, 1, 1, N));
% p_train_cnn =  double(p_train_cnn)';
% p_test_cnn  =  double(p_test_cnn )';

train_cnn = predict(net_cnn, p_train_cnn);
tic;
test_cnn = predict(net_cnn, p_test_cnn );
t4 = toc;
train_cnn = train_cnn';
test_cnn = test_cnn';

% Y_p = (test_grnn + test_bp  + test_f' + test_cnn)./4;
% T_ppp = mapminmax('reverse', Y_p, ps_output);




% X_stack = [train_grnn;train_bp;train_cnn;train_f'];
% X_stack_test = [test_grnn;test_bp;test_cnn;test_f'];
% X_stack = [train_grnn;train_bp;train_f'];
% X_stack_test = [test_grnn;test_bp;test_f'];
X_stack = [train_cnn;train_grnn;train_bp;train_rbf];
X_stack_test = [test_cnn;test_grnn;test_bp;test_rbf];
% net_stack = newff(X_stack,t_train, [10]); % 创建stacking元模型
% 
% % view(net_stack);
% %%  参数设置
% % net_stack.trainParam.goal = 1e-10;
% % net_stack.trainParam.lr = 0.000001;
% 
% net_stack = train(net_stack, X_stack, t_train); % 训练元模型





% T_test_grnn = mapminmax('reverse', test_grnn, ps_output);
% T_test_bp = mapminmax('reverse', test_bp, ps_output);
% 
% T_test_f = mapminmax('reverse', test_f', ps_output);
% T_test_cnn = mapminmax('reverse', test_cnn, ps_output);
% T_p = (T_test_f + T_test_bp + T_test_grnn + T_test_cnn) ./4;
% 



% trees = 100;                                      % 决策树数目
% leaf  = 5;                                        % 最小叶子数
% OOBPrediction = 'on';                             % 打开误差图
% OOBPredictorImportance = 'on';                    % 计算特征重要性
% Method = 'regression';                            % 分类还是回归
% net_stack = TreeBagger(trees, X_stack', t_train, 'OOBPredictorImportance', OOBPredictorImportance,...
%       'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
% importance = net_stack.OOBPermutedPredictorDeltaError;  % 重要性
%                      % 径向基函数的扩展速度
% t_sim1 = predict(net_stack,X_stack'); % 元模型对组合特征预测
% t_sim2 = predict(net_stack,X_stack_test'); % 元模型对组合特征预测
% t_sim1 = t_sim1';
% t_sim2 = t_sim2';


% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Fitting Network
hiddenLayerSize = 10;
net_stack = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net_stack.divideParam.trainRatio = 100/100;


% Train the Network
[net_stack,tr] = train(net_stack,X_stack,t_train);

% Test the Network


% t_sim1 = sim(net_stack, p_train');
% t_sim2 = sim(net_stack, p_test');
% t_sim1 = t_sim1';
% t_sim2 = t_sim2';
t_sim1 = net_stack(X_stack);
tic;
t_sim2 = net_stack(X_stack_test);
t5 = toc;
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  均方根误差RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);
disp(['STACK训练集数据的RMSE为：', num2str(error1)])
disp(['STACK测试集数据的RMSE为：', num2str(error2)])
%%  相关指标计算
% R2
time = t1 + t2 + t3 + t4 + t5;
% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['STACK训练集数据的MAE为：', num2str(mae1)])
disp(['STACK测试集数据的MAE为：', num2str(mae2)])

error_BP1 = abs(T_sim1 - T_train);
error_baifenbi1 = abs(T_sim1 - T_train) ./ T_train;
error_BP2 = abs(T_sim2 - T_test);
error_baifenbi2 = abs(T_sim2 - T_test) ./ T_test;
error_baifenbi1 = sort(error_baifenbi1);
error_baifenbi2 = sort(error_baifenbi2);
mdape1 = (error_baifenbi1(4866) + error_baifenbi1(4867))/2*100;
mdape2 = (error_baifenbi2(540) + error_baifenbi2(541))/2*100;
disp(['STACK训练集数据的MdAPE为：', num2str(mdape1)])
disp(['STACK测试集数据的MdAPE为：', num2str(mdape2)])
% MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% disp(['BP训练集数据的MBE为：', num2str(mbe1)])
% disp(['BP测试集数据的MBE为：', num2str(mbe2)])
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['STACK训练集数据的R2为：', num2str(R1)])
disp(['STACK测试集数据的R2为：', num2str(R2)])

%%  查看网络结构
% view(net_grnn)
error_BP = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
disp(['MAXerr：', num2str(max(error_BP))])
disp(['Minerr：', num2str(min(error_BP))])
disp(['MRE：', num2str(mean(100*error_baifenbi))])
% save A_Me T_sim1 T_train T_sim2 T_test error_BP

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
% disp(['BP训练集数据的RMSE为：', num2str(error1)])
% disp(['BP测试集数据的RMSE为：', num2str(error2)])
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
% error_stack = abs(T_sim2 - T_test);
% error_baifenbi = abs(T_sim2 - T_test) ./ T_test;
% 
% %%  查看网络结构
% % view(net_grnn)
% 
% 
% 
% figure
% plot(1: N, error_stack, 'b-o', 'LineWidth', 1);
% legend('误差绝对值');
% xlabel('测试集样本编号');
% ylabel('误差绝对值 / K');
% string = {'stack测试集样本误差绝对值'};
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
% string = {'stack测试集样本误差百分比比值'};
% title(string);
% xlim([1, N]);
% grid;
% 
% figure
% plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
% legend('真实值', '预测值');
% xlabel('预测样本');
% ylabel('预测结果 / K');
% string = {'stack训练集预测结果对比'; ['RMSE=' num2str(error1)]};
% title(string);
% xlim([1, M]);
% grid;
% 
% figure
% plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果 / K')
% string = {'stack测试集预测结果对比'; ['RMSE=' num2str(error2)]};
% title(string)
% xlim([1, N])
% grid
% 
% %%  相关指标计算
% % R2
% R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
% R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
% disp(['stack训练集数据的R2为：', num2str(R1)])
% disp(['stack测试集数据的R2为：', num2str(R2)])
% % MAE
% mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
% mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
% disp(['stack训练集数据的MAE为：', num2str(mae1)])
% disp(['stack测试集数据的MAE为：', num2str(mae2)])
% % MBE
% mbe1 = sum(T_sim1 - T_train) ./ M ;
% mbe2 = sum(T_sim2 - T_test ) ./ N ;
% disp(['stack训练集数据的MBE为：', num2str(mbe1)])
% disp(['stack测试集数据的MBE为：', num2str(mbe2)])
% %%  绘制散点图
% sz = 25;
% c = 'b';
% 
% figure
% scatter(T_train, T_sim1, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('stack训练集真实值');
% ylabel('stack训练集预测值 / K');
% xlim([min(T_train) max(T_train)])
% ylim([min(T_sim1) max(T_sim1)])
% title('stack训练集预测值 vs. 训练集真实值')
% 
% figure
% scatter(T_test, T_sim2, sz, c)
% hold on
% plot(xlim, ylim, '--k')
% xlabel('stack测试集真实值');
% ylabel('stack测试集预测值 / K');
% xlim([min(T_test) max(T_test)])
% ylim([min(T_sim2) max(T_sim2)])
% title('stack测试集预测值 vs. 测试集真实值')
