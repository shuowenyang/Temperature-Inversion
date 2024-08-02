clc
clear all
close all
%% 加载神经网络的训练样本 测试样本每列一个样本 输入P 输出T，T是标签
%样本数据就是前面问题描述中列出的数据
%epochs是计算时根据输出误差返回调整神经元权值和阀值的次数
load data
% 初始隐层神经元个数
hiddennum=31;
% 输入向量的最大值和最小值
threshold=[0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1];
inputnum=size(P,1); % 输入层神经元个数
outputnum=size(T,1); % 输出层神经元个数
w1num=inputnumhiddennum; % 输入层到隐层的权值个数
w2num=outputnumhiddennum;% 隐层到输出层的权值个数
N=w1num+hiddennum+w2num+outputnum; %待优化的变量的个数

%% 定义遗传算法参数
NIND=40; %个体数目
MAXGEN=50; %最大遗传代数
PRECI=10; %变量的二进制位数
GGAP=0.95; %代沟
px=0.7; %交叉概率
pm=0.01; %变异概率
trace=zeros(N+1,MAXGEN); %寻优结果的初始值

FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)]; %区域描述器
Chrom=crtbp(NIND,PRECI*N); %初始种群
%% 优化
gen=0; %代计数器
X=bs2rv(Chrom,FieldD); %计算初始种群的十进制转换
ObjV=Objfun(X,P,T,hiddennum,P_test,T_test); %计算目标函数值
while gen<MAXGEN
fprintf('%d\n',gen)
FitnV=ranking(ObjV); %分配适应度值
SelCh=select('sus',Chrom,FitnV,GGAP); %选择，随机遍历抽样
SelCh=recombin('xovsp',SelCh,px); %重组，单点交叉
SelCh=mut(SelCh,pm); %变异
X=bs2rv(SelCh,FieldD); %子代个体的十进制转换
ObjVSel=Objfun(X,P,T,hiddennum,P_test,T_test); %计算子代的目标函数值
[Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); %重插入子代到父代，得到新种群，注意插入后新种群与老种群的规模是一样的
%代沟只是说选择子种群的时候是选择95%的个体作为待插入的子种群
%1，父代chrome和子代selch中的子种群个数都是1，1，基于适应度的选择，子代代替父代中适应度最小的个体
X=bs2rv(Chrom,FieldD);%插入完成后，重新计算个体的十进制值
gen=gen+1; %代计数器增加
%获取每代的最优解及其序号，Y为最优解,I为个体的序号
[Y,I]=min(ObjV);%Objv是目标函数值，也就是预测误差的范数
trace(1:N,gen)=X(I,:); %记下每代个体的最优值，即各个权重值
trace(end,gen)=Y; %记下每代目标函数的最优值，即预测误差的范数
end
%% 画进化图
figure(1);
plot(1:MAXGEN,trace(end,:));
grid on
xlabel('遗传代数')
ylabel('误差的变化')
title('进化过程')
bestX=trace(1:end-1,end);%注意这里仅是记录下了最优的初始权重，训练得到的最终的网络的权值并未记录下来
bestErr=trace(end,end);
fprintf(['最优初始权值和阈值:\nX=',num2str(bestX'),'\n最小误差err=',num2str(bestErr),'\n'])
%% 比较优化前后的训练&测试
callbackfun

% 子函数：
function Obj=Objfun(X,P,T,hiddennum,P_test,T_test)
%% 用来分别求解种群中各个个体的目标值
%% 输入
% X：所有个体的初始权值和阈值
% P：训练样本输入
% T：训练样本输出
% hiddennum：隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
%% 输出
% Obj：所有个体的预测样本的预测误差的范数
%这个函数的目的就是用种群中所有个体所代表的神经网络的初始权重值去进行网络的训练，训练次数是1000次，然
%后得出所有个体作为初始权重训练网络1000次所得出的预测误差，也就是这里的obj，返回到原函数中，迭代maxgen=50次
%记录下每一代的最优权重值和最优目标值(最小误差值)
[M,N]=size(X);
Obj=zeros(M,1);
for i=1:M%M是40，即有40个个体，每个个体就是一次初始权重，在BPfun中用每个个体作为初始值去进行了1000次的训练
Obj(i)=BPfun(X(i,:),P,T,hiddennum,P_test,T_test);%Obj是一个40*1的向量，每个值对应的是一个个体作为初始权重值去进行训练
%网络1000次得出来的误差
end

function err=BPfun(x,P,T,hiddennum,P_test,T_test)
%% 训练&测试BP网络
%% 输入
% x：一个个体的初始权值和阈值
% P：训练样本输入
% T：训练样本输出
% hiddennum：隐含层神经元数
% P_test:测试样本输入
% T_test:测试样本期望输出
%% 输出
% err：预测样本的预测误差的范数
%用每一个个体的初始权值去训练1000次
inputnum=size(P,1); % 输入层神经元个数
outputnum=size(T,1); % 输出层神经元个数
%% 新建BP网络
%神经网络的隐含层神经元的传递函数采用S型正切函数tansing（），输出层神经元的函数采用S型对数函数logsig（）
net=newff(minmax,[hiddennum,outputnum],{'tansig','logsig'},'trainlm');
%% 设置网络参数：训练次数为1000，训练目标为0.01，学习速率为0.1
net.trainParam.epochs=1000;%允许最大训练次数，实际这个网络训练到迭代次数是3时就已经到达要求结束了
net.trainParam.goal=0.01;%训练目标最小误差，应该是mean square error， 均方误差，就是网络输出和目标值的差的平方再求平均值
LP.lr=0.1;%学习速率学习率的作用是不断调整权值阈值。w(n+1)=w(n)+LP.lr*(d(n)-y(n))x(n),d(n)是期望的相应，y(n)是
%量化的实际响应，x(n)是输入向量，如果d(n)与y(n)相等的话，则w(n+1)=w(n),这里是指输入到隐含层的调整方式
%隐含层到输出层的调整 Iout(j)=1/(1+exp(-I(j)));
%dw2=eIout;db2=e’;w2=w2_1+xitedw2’;e是错误值
%b2=b2_1+xitedb2’;xite是学习率
%对于traingdm等函数建立的BP网络，学习速率一般取0.01-0.1之间。
net.trainParam.show=NaN;
% net.trainParam.showwindow=false; %高版MATLAB
%% BP神经网络初始权值和阈值
w1num=inputnumhiddennum; % 输入层到隐层的权值个数
w2num=outputnumhiddennum;% 隐层到输出层的权值个数
w1=x(1:w1num); %初始输入层到隐层的权值
B1=x(w1num+1:w1num+hiddennum); %初始隐层阈值
w2=x(w1num+hiddennum+1:w1num+hiddennum+w2num); %初始隐层到输出层的阈值
B2=x(w1num+hiddennum+w2num+1:w1num+hiddennum+w2num+outputnum); %输出层阈值
net.iw{1,1}=reshape(w1,hiddennum,inputnum);%输入到隐藏层的权重
net.lw{2,1}=reshape(w2,outputnum,hiddennum);%隐藏到输出层的权重
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=reshape(B2,outputnum,1);
%% 训练网络以
net=train(net,P,T);
%% 测试网络
Y=sim(net,P_test);%测试样本的仿真结果
err=norm(Y-T_test);%测试样本的仿真误差

% callbackfun函数，比较实用遗传算法和不使用遗传算法优化的结果对比
clc
%% 不使用遗传算法
%% 使用随机权值和阈值
% P：训练样本输入
% T：训练样本标签
% P_test:测试样本输入
% T_test:测试样本期望输出

inputnum=size(P,1); % 输入层神经元个数
outputnum=size(T,1); % 输出层神经元个数
%% 新建BP网络
net=newff(minmax,[hiddennum,outputnum],{'tansig','logsig'},'trainlm');
%% 设置网络参数：训练次数为1000，训练目标为0.01，学习速率为0.1
net.trainParam.epochs=1000;
net.trainParam.goal=0.01;
LP.lr=0.1;
%% 训练网络以
net=train(net,P,T);
%% 测试网络
disp(['1、使用随机权值和阈值 '])
disp('测试样本预测结果：')
Y1=sim(net,P_test)%测试样本的网络仿真输出
err1=norm(Y1-T_test); %测试样本的仿真误差
err11=norm(sim(net,P)-T); %训练样本的仿真误差
disp(['测试样本的仿真误差:',num2str(err1)])
disp(['训练样本的仿真误差:',num2str(err11)])

%% 使用遗传算法
%% 使用优化后的权值和阈值，利用遗传算法得出来的最优的初始权重和阈值去进行网络的初始化
inputnum=size(P,1); % 输入层神经元个数
outputnum=size(T,1); % 输出层神经元个数
%% 新建BP网络
net=newff(minmax,[hiddennum,outputnum],{'tansig','logsig'},'trainlm');
%% 设置网络参数：训练次数为1000，训练目标为0.01，学习速率为0.1
net.trainParam.epochs=1000;
net.trainParam.goal=0.01;
LP.lr=0.1;
%% BP神经网络初始权值和阈值
w1num=inputnumhiddennum; % 输入层到隐层的权值个数
w2num=outputnumhiddennum;% 隐层到输出层的权值个数
w1=bestX(1:w1num); %初始输入层到隐层的权值
B1=bestX(w1num+1:w1num+hiddennum); %初始隐层阈值
w2=bestX(w1num+hiddennum+1:w1num+hiddennum+w2num); %初始隐层到输出层的阈值
B2=bestX(w1num+hiddennum+w2num+1:w1num+hiddennum+w2num+outputnum); %输出层阈值
net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=reshape(B2,outputnum,1);
%% 训练网络以
net=train(net,P,T);
%% 测试网络
disp(['2、使用优化后的权值和阈值'])
disp('测试样本预测结果：')
Y2=sim(net,P_test)%测试样本的仿真输出
err2=norm(Y2-T_test);%测试样本的仿真误差
err21=norm(sim(net,P)-T);%训练样本的仿真误差
disp(['测试样本的仿真误差:',num2str(err2)])
disp(['训练样本的仿真误差:',num2str(err21)])
end
end