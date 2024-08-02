
 
%% ���³���Ϊ������չ���GRNN��BP�Ƚ� ��Ҫload chapter8.1���������
close all
clear all
load data_fangzhen.mat;


% �������ѵ�����Ͳ��Լ�

P_train = [];
T_train = [];
P_test = [];
T_test = [];
for i = 1:4
    temp_input = data_output((i-1)*310+1:i*310,:);
    temp_output = data_output_K((i-1)*310+1:i*310,:);
    n = randperm(310);
    % ѵ��������20������
    P_train = [P_train temp_input(n(1:248),:)'];
    T_train = [T_train temp_output(n(1:248),:)'];
    % ���Լ�����2������
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
% ѵ������%����ѵ������

net.trainParam.lr=0.01;               %ѧϰ��
net.trainparam.show=40;                %ÿѵ��400��չʾһ�ν��
net.trainParam.goal=0.01;             %Ŀ�����
net.trainparam.epochs=2000;            %���ѵ������
net.trainParam.showWindow=1;            %BPѵ������
% net.divideParam.trainRatio=0.8;         % ����ѵ�������ݱ���
% net.divideParam.valRatio=0.2;           % ������֤����ϵ����ݱ���
% net.divideParam.testRatio=0;            % ע��Ҫ�ص���������ռ��

%����TRAINLM�㷨ѵ��BP����
net_bp=train(net_bp,p_train,t_train);
%%  �������
t_sim1 = sim(net_bp, p_train);
t_sim2 = sim(net_bp, p_test );
%%  ���ݷ���һ��
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
% error00 = sum((T_sim1 - T_train).^2) ./ M
% mse1 = mse(T_sim1, T_train)
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

error_BP = abs(T_sim2 - T_test);
error_baifenbi = abs(T_sim2 - T_test) ./ T_test;

%%  �鿴����ṹ
% view(net_grnn)

figure
plot(1: N, error_BP, 'b-o', 'LineWidth', 1);
legend('������ֵ');
xlabel('���Լ��������');
ylabel('������ֵ / K');
string = {'BP���Լ�����������ֵ'};
title(string);
xlim([1, N]);
grid;


figure
plot(1: N, error_baifenbi, 'r-*', 'LineWidth', 1);
legend('���ٷֱȱ�ֵ');
xlabel('���Լ��������');
ylabel('���ٷֱȱ�ֵ');
string = {'BP���Լ��������ٷֱȱ�ֵ'};
title(string);
xlim([1, N]);
grid;

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1);
legend('��ʵֵ', 'Ԥ��ֵ');
xlabel('Ԥ������');
ylabel('Ԥ���� / K');
string = {'BPѵ����Ԥ�����Ա�'; ['RMSE=' num2str(error1)]};
title(string);
xlim([1, M]);
grid;

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ���� / K')
string = {'BP���Լ�Ԥ�����Ա�'; ['RMSE=' num2str(error2)]};
title(string)
xlim([1, N])
grid

%%  ���ָ�����
% R2
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2)^2 / norm(T_test  - mean(T_test ))^2;
disp(['BPѵ�������ݵ�R2Ϊ��', num2str(R1)])
disp(['BP���Լ����ݵ�R2Ϊ��', num2str(R2)])
% MAE
mae1 = sum(abs(T_sim1 - T_train)) ./ M ;
mae2 = sum(abs(T_sim2 - T_test )) ./ N ;
disp(['BPѵ�������ݵ�MAEΪ��', num2str(mae1)])
disp(['BP���Լ����ݵ�MAEΪ��', num2str(mae2)])
% MBE
mbe1 = sum(T_sim1 - T_train) ./ M ;
mbe2 = sum(T_sim2 - T_test ) ./ N ;
disp(['BPѵ�������ݵ�MBEΪ��', num2str(mbe1)])
disp(['BP���Լ����ݵ�MBEΪ��', num2str(mbe2)])
%%  ����ɢ��ͼ
sz = 25;
c = 'b';

figure
scatter(T_train, T_sim1, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('BPѵ������ʵֵ');
ylabel('BPѵ����Ԥ��ֵ / K');
xlim([min(T_train) max(T_train)])
ylim([min(T_sim1) max(T_sim1)])
title('BPѵ����Ԥ��ֵ vs. ѵ������ʵֵ')

figure
scatter(T_test, T_sim2, sz, c)
hold on
plot(xlim, ylim, '--k')
xlabel('BP���Լ���ʵֵ');
ylabel('BP���Լ�Ԥ��ֵ / K');
xlim([min(T_test) max(T_test)])
ylim([min(T_sim2) max(T_sim2)])
title('BP���Լ�Ԥ��ֵ vs. ���Լ���ʵֵ')