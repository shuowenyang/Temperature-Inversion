% 生成示例数据
clc;clear;
numSamples = 1000;
inputSize = 6;
outputSize = 1;

X_train = randn([inputSize, numSamples]);
Y_train = randn([outputSize, numSamples]);

X_val = randn([inputSize, numSamples]);
Y_val = randn([outputSize, numSamples]);

% 构建LSTM网络
layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(10)
    fullyConnectedLayer(outputSize)
    regressionLayer];

% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {X_val, Y_val}, ...
    'Plots', 'training-progress');

% 训练网络

net = trainNetwork(X_train, Y_train, layers, options);

% 验证网络
Y_val_pred = predict(net, X_val);

% 展示结果
figure;
plot(Y_val, Y_val_pred, '.');
xlabel('True');
ylabel('Predicted');
title('Validation Results');

% 测试数据
X_test = randn([inputSize, numSamples]);
Y_test_pred = predict(net, X_test);

% 展示预测结果
figure;
plot(1:numSamples, Y_test_pred);
xlabel('Sample');
ylabel('Predicted Value');
title('Test Predictions');
