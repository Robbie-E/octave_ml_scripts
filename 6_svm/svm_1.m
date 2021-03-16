% Load from ex6data1, use linear decision boundary 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
plotData(X, y);

% Choose C, train SVM with linear decision boundary
C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);