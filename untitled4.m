%% 1.数据预处理
% data = readmatrix('data_fangzhen229.mat');  %读取数据文件
clc;clear;
load data_fangzhen229.mat;
rng(1);%设置随机种子

data = data_output;
% 分割数据为训练集和测试集
% trainRatio = 0.7; % 70% 用于训练，其余的用于测试
% trainCount = floor(size(data, 1) * trainRatio);
% trainData = data(1:trainCount, :);
% testData = data(trainCount+1:end, :);

% % 分割输入特征和目标变量
% XTrain = trainData(:, 1:end-1)';
% YTrain = trainData(:, end)';
% XTest = testData(:, 1:end-1)';
% YTest = testData(:, end)';

XTrain =[];
YTrain =[];
XTest = [];
YTest = [];
for i = 1:4
    temp_input = data((i-1)*110+1:i*110,1:6);
    temp_output = data((i-1)*110+1:i*110,7);
    n = randperm(110);
    % 训练集——20个样本
    XTrain = [XTrain;temp_input(n(1:88),1:6)];
    % [XTrain temp_input(:,:,n(1:276))];
    YTrain = [YTrain;temp_output(n(1:88))];
    % 测试集——2个样本
    XTest = [XTest;temp_input(n(89:110),1:6)];
    YTest = [YTest;temp_output(n(89:110))];
end
XTrain = XTrain';
XTest = XTest';
YTrain = YTrain';
YTest = YTest';

%% 2. 数据归一化
method=@mapminmax;
[XTrainMap,inputps]=method(XTrain);
XTestMap=method('apply',XTest,inputps);
[YTrainMap,outputps]=method(YTrain);
YTestMap=method('apply',YTest,outputps);

%% 3. 数据转换
XTrainMapD=reshape(XTrain,[size(XTrain,1),length(XTrain)]);%训练集输入
XTestMapD =reshape(XTest,  [size(XTest,1),length(XTest)]); %测试集输入

inputSize = 6;
outputSize = 1;
%% 4.构建 CNN 模型
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(128)
    fullyConnectedLayer(outputSize)
    % sigmoidLayer
    regressionLayer];



% 显示层信息

analyzeNetwork(layers)

%% 5.指定训练选项
options = trainingOptions('adam', ...%求解器，'sgdm'（默认） | 'rmsprop' | 'adam'
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...  %梯度极限
    'MaxEpochs',500, ...%最大迭代次数
    'InitialLearnRate', 0.01, ...%初始化学习速率
    'ValidationFrequency',10, ...%验证频率，即每间隔多少次迭代进行一次验证
    'MiniBatchSize',128, ...
    'LearnRateSchedule','piecewise', ...%是否在一定迭代次数后学习速率下降
    'LearnRateDropFactor',0.9, ...%学习速率下降因子
    'LearnRateDropPeriod',10, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XTestMapD,YTestMap},...
    'Verbose',true, ...
    'Plots','training-progress');%显示训练过程

% 训练模型
net = trainNetwork(XTrainMapD,YTrainMap,layers,options);

%% 6.对测试集进行预测
YPred = predict(net,XTestMapD); 
% 反归一化
foreData=double(method('reverse',double(YPred'),outputps));


%% 7.对训练集进行拟合
YpredTrain = predict(net,XTrainMapD); 
% 反归一化
foreDataTrain=double(method('reverse',double(YpredTrain'),outputps));

%% 8. 训练集预测结果对比
figure('Color','w')
plot(foreDataTrain,'-','Color',[255 0 0]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[250 0 0]./255)
hold on 
plot(YTrain,'-','Color',[150 150 150]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[150 150 150]./255)
legend('CNN训练集预测值','真实值')
xlabel('预测样本')
ylabel('预测结果')
xlim([1, length(foreDataTrain)])
grid
ax=gca;hold on


%% 9. 测试集预测结果对比
figure('Color','w')
plot(foreData,'-','Color',[0 0 255]./255,'linewidth',1,'Markersize',5,'MarkerFaceColor',[0 0 255]./255)
hold on 
plot(YTest,'-','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[0 0 0]./255)
legend('CNN测试集预测值','真实值')
xlabel('预测样本')
ylabel('预测结果')
xlim([1, length(foreData)])
grid
ax=gca;hold on

