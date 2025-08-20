clc; clear; close all;

%% Load dataset
filename = 'C:\Users\Hp\OneDrive\Desktop\Chemical_Reactor_pH_Dataset_2000.csv';
data = readtable(filename);

% Inspect variable names (optional)
disp('Variable names in table:');
disp(data.Properties.VariableNames);

%% Extract input (first column) and output (second column)
X = data{:,1};   % First column = Concentration
Y = data{:,2};   % Second column = Reactor pH

%% Store mean and std for denormalization
meanY = mean(Y);
stdY = std(Y);

%% Manual Normalization (since normalize() is unavailable)
Xn = (X - min(X)) / (max(X) - min(X));   % Min-Max scaling [0,1]
Yn = (Y - meanY) / stdY;                 % Z-score normalization

%% Split dataset into train/validation/test
n = length(Xn);
idx = randperm(n);
trainEnd = round(0.7*n);
valEnd = round(0.85*n);

trainInd = idx(1:trainEnd);
valInd   = idx(trainEnd+1:valEnd);
testInd  = idx(valEnd+1:end);

inputTrain = Xn(trainInd)';
targetTrain = Yn(trainInd)';   
inputVal   = Xn(valInd,:)';
targetVal  = Yn(valInd)';
inputTest  = Xn(testInd,:)';
targetTest = Yn(testInd,:)';

%% Define Neural Network (SISO)
hiddenLayerSizes = [20 10];   % Example architecture
net = feedforwardnet(hiddenLayerSizes,'traingd');  % Gradient Descent
net.trainParam.epochs = 20000;
net.trainParam.lr = 0.001;       % Learning rate
net.trainParam.goal = 1e-6;     % MSE goal

% Custom data division
net.divideFcn = 'divideind';  
net.divideParam.trainInd = 1:length(inputTrain);
net.divideParam.valInd   = length(inputTrain)+1:length([inputTrain inputVal]);
net.divideParam.testInd  = [];

% Combine train + validation
inputAll = [inputTrain inputVal];
targetAll = [targetTrain targetVal];

%% Train the network
net = train(net, inputAll, targetAll);

%% Test prediction
Ypred_norm = net(inputTest);
Ypred = Ypred_norm * stdY + meanY;   % Denormalize
actual = Y(testInd);

%% Calculate Metrics
mse_error = mean((actual - Ypred').^2);
rmse_error = sqrt(mse_error);
mae_error = mean(abs(actual - Ypred'));
R2 = 1 - sum((actual - Ypred').^2) / sum((actual - mean(actual)).^2);

fprintf('MSE: %.4f\n', mse_error);
fprintf('RMSE: %.4f\n', rmse_error);
fprintf('MAE: %.4f\n', mae_error);
fprintf('R²: %.4f\n', R2);

%% Normalized Comparison
t = targetTest;
t_hat = Ypred_norm;
figure;
plot(t,'b','LineWidth',1.5); hold on;
plot(t_hat,'r--','LineWidth',1.5);
legend('Actual','Predicted');
xlabel('Sample index'); ylabel('Normalized pH');
title('Prediction (Normalized)');
grid on;

