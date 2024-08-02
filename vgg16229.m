%% 1.数据预处理
% data = readmatrix('data_fangzhen229.mat');  %读取数据文件
clear all;
load data_tu_0229.mat;
rng(1);%设置随机种子

data = tu;
% 分割数据为训练集和测试集
% trainRatio = 0.7; % 70% 用于训练，其余的用于测试
% trainCount = floor(size(data, 3) * trainRatio);
% trainData = data(:, :, 1:trainCount);
% testData = data(:, :, trainCount+1:end);
XTrain =[];
YTrain =[];
XTest = [];
YTest = [];
for i = 1:5
    temp_input = data(:,:,(i-1)*368+1:i*368);
    temp_output = data_output_K((i-1)*368+1:i*368,:);
    n = randperm(368);
    % 训练集——20个样本
    XTrain = cat(3,XTrain, temp_input(:,:,n(1:276)));
    % [XTrain temp_input(:,:,n(1:276))];
    YTrain = cat(1,YTrain, temp_output(n(1:276)));
    % 测试集——2个样本
    XTest = cat(3,XTest, temp_input(:,:,n(277 : 368)));
    YTest = cat(1,YTest, temp_output(n(277 : 368)));
end
YTrain = YTrain';
YTest = YTest';
% 分割输入特征和目标变量
% XTrain = trainData(:, :, 1:trainCount);
% YTrain = data_output_K(1:trainCount,:)';
% XTest = testData;
% YTest = data_output_K(trainCount + 1 : end,:)';
% 
% %% 2. 数据归一化
method=@mapminmax;
% 
% [XTrainMap,inputps]=method(XTrain(:,:,i));
% XTestMap=method('apply',XTest(:,:,i),inputps);
% 
XTrainMap = XTrain;
XTestMap = XTest;
[YTrainMap,outputps]=method(YTrain);
YTestMap=method('apply',YTest,outputps);
% YTrainMap = YTrain;
% YTestMap = YTest;
%% 3. 数据转换
XTrainMapD=reshape(XTrain, [size(XTrain,1),size(XTrain,2),1,length(XTrain)]);%训练集输入
XTestMapD =reshape(XTest,  [size(XTest,1),size(XTrain,2),1,length(XTest)]); %测试集输入
% XTrainMapD=XTrain;
% XTestMapD =XTest;
%% 4.构建 CNN 模型
% 创建层


% layers = [
%     imageInputLayer([size(XTrain, 1),1 1]) % 输入层
%     convolution2dLayer([3,1],64,'Stride',1,'Padding',1) % 卷积层
%     batchNormalizationLayer
%     reluLayer %ReLU激活函数层
%     fullyConnectedLayer(1) % 全连接层
%     regressionLayer]; % 回归层
% 
% % 显示层信息
% analyzeNetwork(layers)

layers = [
    % 卷积层1
    imageInputLayer([size(XTrain, 1),size(XTrain, 2) 1])
    
    convolution2dLayer([2 2],64,'Padding','same')
    reluLayer()
    convolution2dLayer([2 2],64,'Padding','same')
    reluLayer()  
    % maxPooling2dLayer([1 1],'Stride',1)
    maxPooling2dLayer(1,'Stride',1)
    % 卷积层2
    convolution2dLayer([2 2],128,'Padding','same') 
    reluLayer()
    convolution2dLayer([2 2],128,'Padding','same')
    reluLayer()
    maxPooling2dLayer(1,'Stride',1)
    
    % 卷积层3 
    convolution2dLayer([2 2],256,'Padding','same')
    reluLayer()
    convolution2dLayer([2 2],256,'Padding','same') 
    reluLayer()
    convolution2dLayer([2 2],256,'Padding','same')
    reluLayer()
    maxPooling2dLayer(1,'Stride',1)

    % % 卷积层4
    % convolution2dLayer([3 3],512,'Padding','same')
    % reluLayer()
    % convolution2dLayer([3 3],512,'Padding','same')
    % reluLayer()
    % convolution2dLayer([3 3],512,'Padding','same') 
    % reluLayer()  
    % maxPooling2dLayer(1,'Stride',1)
    % 
    % % 卷积层5
    % convolution2dLayer([3 3],512,'Padding','same')
    % reluLayer() 
    % convolution2dLayer([3 3],512,'Padding','same')
    % reluLayer()
    % convolution2dLayer([3 3],512,'Padding','same')
    % reluLayer()
    % maxPooling2dLayer(1,'Stride',1)
    
    fullyConnectedLayer(1)
    regressionLayer]; % 回归层

% layers = vgg16('Weights','none')


% 显示层信息
% 
% analyzeNetwork(layers)
% disp(layers)

%% 5.指定训练选项
options = trainingOptions('sgdm', ...%求解器，'sgdm'（默认） | 'rmsprop' | 'adam'
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...  %梯度极限
    'MaxEpochs',200, ...%最大迭代次数
    'InitialLearnRate', 0.001, ...%初始化学习速率
    'ValidationFrequency',10, ...%验证频率，即每间隔多少次迭代进行一次验证
    'MiniBatchSize',256, ...
    'LearnRateSchedule','piecewise', ...%是否在一定迭代次数后学习速率下降
    'LearnRateDropFactor',0.7, ...%学习速率下降因子
    'LearnRateDropPeriod',10, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'ValidationData',{XTestMapD,YTestMap'},...
    'Verbose',true, ...
    'Plots','training-progress');%显示训练过程

% 训练模型
net = trainNetwork(XTrainMapD,YTrainMap',layers,options);

%% 6.对测试集进行预测
YPred = predict(net,XTestMapD); 
% 反归一化
foreData=double(method('reverse',double(YPred'),outputps));

errorp = YTest-foreData;
%% 7.对训练集进行拟合
YpredTrain = predict(net,XTrainMapD); 
% 反归一化
foreDataTrain=double(method('reverse',double(YpredTrain'),outputps));
errorT = YTrain-foreDataTrain;
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

