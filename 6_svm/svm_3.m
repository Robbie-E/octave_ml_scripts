% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

% Try different SVM Parameters, find optimal parameters
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);