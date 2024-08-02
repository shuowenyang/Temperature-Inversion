%% Matlab神经网络43个案例分析

% GRNN的数据预测—基于广义回归神经网络的货运量预测
% by 王小川(@王小川_matlab)
% http://www.matlabsky.com
% Email:sina363@163.com
% http://weibo.com/hgsz2003
 
%% 清空环境变量
clc;
clear all
close all
load data_fangzhen.mat;


% 随机产生训练集和测试集

p_train = [];
t_train = [];
p_test = [];
t_test = [];
% for i = 1:4
%     temp_input = data_output((i-1)*310+1:i*310,:);
%     temp_output = data_output_K((i-1)*310+1:i*310,:);
%     n = randperm(310);
%     % 训练集——20个样本
%     p_train = [p_train temp_input(n(1:248),:)'];
%     t_train = [t_train temp_output(n(1:248),:)'];
%     % 测试集——2个样本
%     p_test = [p_test temp_input(n(249 : 310),:)'];
%     t_test = [t_test temp_output(n(249 : 310),:)'];
% end
p_train=data_output(1:992,:);
t_train=data_output_K(1:992,:);
p_test=data_output(993:1240,:);
t_test=data_output_K(993:1240,:);
%% 交叉验证
desired_spread=[];
mse_max=10e20;
desired_input=[];
desired_output=[];
result_perfp=[];
indices = crossvalind('Kfold',length(p_train),5);
h=waitbar(0,'正在寻找最优化参数....');
k=1;
for i = 1:5
    perfp=[];
    disp(['以下为第',num2str(i),'次交叉验证结果'])
    test = (indices == i); train = ~test;
    p_cv_train=p_train(train,:);
    t_cv_train=t_train(train,:);
    p_cv_test=p_train(test,:);
    t_cv_test=t_train(test,:);
    p_cv_train=p_cv_train';
    t_cv_train=t_cv_train';
    p_cv_test= p_cv_test';
    t_cv_test= t_cv_test';
    [p_cv_train,ps]=mapminmax(p_cv_train);
    [t_cv_train,ts]=mapminmax(t_cv_train);
    p_cv_test=mapminmax(p_cv_test,ps);
    % p_cv_test=tramnmx(p_cv_test,minp,maxp);
    for spread=0.1:0.1:2;
        net=newgrnn(p_cv_train,t_cv_train,spread);
        waitbar(k/80,h);
        disp(['当前spread值为', num2str(spread)]);
        test_Out=sim(net,p_cv_test);
        % test_Out=postmnmx(test_Out,mint,maxt);
        test_Out=mapminmax('reverse',test_Out,ts);
        error=t_cv_test-test_Out;
        disp(['当前网络的mse为',num2str(mse(error))])
        perfp=[perfp mse(error)];
        if mse(error)<mse_max
            mse_max=mse(error);
            desired_spread=spread;
            desired_input=p_cv_train;
            desired_output=t_cv_train;
        end
        k=k+1;
    end
    result_perfp(i,:)=perfp;
end;
close(h)
disp(['最佳spread值为',num2str(desired_spread)])
disp(['此时最佳输入值为'])
desired_input;
disp(['此时最佳输出值为'])
desired_output;
%% 采用最佳方法建立GRNN网络
net=newgrnn(desired_input,desired_output,desired_spread);
p_test=p_test';
% p_test=tramnmx(p_test,minp,maxp);
p_test=mapminmax(p_test,ps);
grnn_prediction_result=sim(net,p_test);
grnn_prediction_result=mapminmax('reverse',grnn_prediction_result,ts);
grnn_error=abs(t_test-grnn_prediction_result');
% disp(['GRNN神经网络预测的误差为',num2str(abs(grnn_error(1))),' ',num2str(abs(grnn_error(2)))])
% disp(['GRNN神经网络预测的误差为',num2str(abs(grnn_error(1)))]);
% save best desired_input desired_output p_test t_test grnn_error ts

