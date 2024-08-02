%% Matlab������43����������

% GRNN������Ԥ�⡪���ڹ���ع�������Ļ�����Ԥ��
% by ��С��(@��С��_matlab)
% http://www.matlabsky.com
% Email:sina363@163.com
% http://weibo.com/hgsz2003
 
%% ��ջ�������
clc;
clear all
close all
load data_fangzhen.mat;


% �������ѵ�����Ͳ��Լ�

p_train = [];
t_train = [];
p_test = [];
t_test = [];
% for i = 1:4
%     temp_input = data_output((i-1)*310+1:i*310,:);
%     temp_output = data_output_K((i-1)*310+1:i*310,:);
%     n = randperm(310);
%     % ѵ��������20������
%     p_train = [p_train temp_input(n(1:248),:)'];
%     t_train = [t_train temp_output(n(1:248),:)'];
%     % ���Լ�����2������
%     p_test = [p_test temp_input(n(249 : 310),:)'];
%     t_test = [t_test temp_output(n(249 : 310),:)'];
% end
p_train=data_output(1:992,:);
t_train=data_output_K(1:992,:);
p_test=data_output(993:1240,:);
t_test=data_output_K(993:1240,:);
%% ������֤
desired_spread=[];
mse_max=10e20;
desired_input=[];
desired_output=[];
result_perfp=[];
indices = crossvalind('Kfold',length(p_train),5);
h=waitbar(0,'����Ѱ�����Ż�����....');
k=1;
for i = 1:5
    perfp=[];
    disp(['����Ϊ��',num2str(i),'�ν�����֤���'])
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
        disp(['��ǰspreadֵΪ', num2str(spread)]);
        test_Out=sim(net,p_cv_test);
        % test_Out=postmnmx(test_Out,mint,maxt);
        test_Out=mapminmax('reverse',test_Out,ts);
        error=t_cv_test-test_Out;
        disp(['��ǰ�����mseΪ',num2str(mse(error))])
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
disp(['���spreadֵΪ',num2str(desired_spread)])
disp(['��ʱ�������ֵΪ'])
desired_input;
disp(['��ʱ������ֵΪ'])
desired_output;
%% ������ѷ�������GRNN����
net=newgrnn(desired_input,desired_output,desired_spread);
p_test=p_test';
% p_test=tramnmx(p_test,minp,maxp);
p_test=mapminmax(p_test,ps);
grnn_prediction_result=sim(net,p_test);
grnn_prediction_result=mapminmax('reverse',grnn_prediction_result,ts);
grnn_error=abs(t_test-grnn_prediction_result');
% disp(['GRNN������Ԥ������Ϊ',num2str(abs(grnn_error(1))),' ',num2str(abs(grnn_error(2)))])
% disp(['GRNN������Ԥ������Ϊ',num2str(abs(grnn_error(1)))]);
% save best desired_input desired_output p_test t_test grnn_error ts

