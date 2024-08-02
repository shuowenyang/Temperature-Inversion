%% 1.数据预处理
% data = readmatrix('data_fangzhen229.mat');  %读取数据文件
load data_fangzhen229.mat;
rng(1);%设置随机种子

data = data_output;
% 分割数据为训练集和测试集
trainRatio = 0.7; % 70% 用于训练，其余的用于测试
trainCount = floor(size(data, 1) * trainRatio);
trainData = data(1:trainCount, :);
testData = data(trainCount+1:end, :);

% 分割输入特征和目标变量
XTrain = trainData(:, 1:end-1)';
YTrain = trainData(:, end)';
XTest = testData(:, 1:end-1)';
YTest = testData(:, end)';

%% 2. 数据归一化
method=@mapminmax;
[XTrainMap,inputps]=method(XTrain);
XTestMap=method('apply',XTest,inputps);
[YTrainMap,outputps]=method(YTrain);
YTestMap=method('apply',YTest,outputps);

%% 3. 数据转换
XTrainMapD=reshape(XTrain,[size(XTrain,1),1,1,length(XTrain)]);%训练集输入
XTestMapD =reshape(XTest,  [size(XTest,1),1,1,length(XTest)]); %测试集输入

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

% layers = [
%     % 卷积层1
%     imageInputLayer([size(XTrain, 1),1 1])
% 
%     convolution2dLayer([3 3],64,'Padding','same')
%     reluLayer()
%     convolution2dLayer([3 3],64,'Padding','same')
%     reluLayer()  
%     maxPooling2dLayer([1 1],'Stride',1)
% 
%     % 卷积层2
%     convolution2dLayer([3 3],128,'Padding','same') 
%     reluLayer()
%     convolution2dLayer([3 3],128,'Padding','same')
%     reluLayer()
%     maxPooling2dLayer([1 1],'Stride',1) 
% 
%     % 卷积层3 
%     convolution2dLayer([3 3],256,'Padding','same')
%     reluLayer()
%     convolution2dLayer([3 3],256,'Padding','same') 
%     reluLayer()
%     convolution2dLayer([3 3],256,'Padding','same')
%     reluLayer()
%     maxPooling2dLayer([1 1],'Stride',1)
% 
%     % 卷积层4
%     convolution2dLayer([3 3],512,'Padding','same')
%     reluLayer()
%     convolution2dLayer([3 3],512,'Padding','same')
%     reluLayer()
%     convolution2dLayer([3 3],512,'Padding','same') 
%     reluLayer()  
%     maxPooling2dLayer([1 1],'Stride',1)
% 
%     % 卷积层5
%     convolution2dLayer([3 3],512,'Padding','same')
%     reluLayer() 
%     convolution2dLayer([3 3],512,'Padding','same')
%     reluLayer()
%     convolution2dLayer([3 3],512,'Padding','same')
%     reluLayer()
%     maxPooling2dLayer([1 1],'Stride',1)
% 
%     fullyConnectedLayer(1)
%     regressionLayer]; % 回归层

% layers = vgg16('Weights','none')


% 显示层信息
% 创建一个ResNet50网络
net5 = resnet50('Weights','none');

% 修改输出层以适应回归任务
numOutputs = 1; % 回归任务只有一个输出
newLayers = [
    imageInputLayer([size(XTrain, 1),1 1],'Name','input')
    fullyConnectedLayer(numOutputs, 'Name', 'fc_final')
    regressionLayer('Name', 'output')];



% 替换网络的输出层

net5 = removeLayers(net5, 'ClassificationLayer_fc1000');
% gapLayer = globalAveragePooling2dLayer('Name', 'global_avg_pool');
% net = replaceLayer(net, 'avg_pool', gapLayer);

net5 = replaceLayer(net5, 'input_1',newLayers(1));
net5 = replaceLayer(net5, 'fc1000', newLayers(2));

net5 = replaceLayer(net5, 'fc1000_softmax', newLayers(3));
% net = replaceLayer(net, 'ClassificationLayer_fc1000', newLayers(3));
% 显示修改后的网络结构
% analyzeNetwork(net);



analyzeNetwork(net5);

%% 5.指定训练选项
   options = trainingOptions('sgdm', ...%求解器，''（默认） | 'rmsprop' | 'adam'
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',Inf, ...  %梯度极限
    'MaxEpochs',500, ...%最大迭代次数
    'InitialLearnRate', 0.001, ...%初始化学习速率
    'ValidationFrequency',10, ...%验证频率，即每间隔多少次迭代进行一次验证
    'MiniBatchSize',64, ...
    'LearnRateSchedule','piecewise', ...%是否在一定迭代次数后学习速率下降
    'LearnRateDropFactor',0.9, ...%学习速率下降因子
    'LearnRateDropPeriod',10, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'ValidationData',{XTestMapD,YTestMap'},...
    'Verbose',true, ...
    'Plots','training-progress');%显示训练过程

% 训练模型
net = trainNetwork(XTrainMapD,YTrainMap',net5,options);
% net = trainNetwork(XTrainMapD,YTrainMap',layers_1,options);
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

