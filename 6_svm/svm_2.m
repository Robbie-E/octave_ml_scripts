%==========================================
% Test Gausssian kernel
% expect 0.324652 with sigma set to 2
x1 = [1 2 1]; x2 = [0 4 -1]; 
sigma = 2;
sim = gaussianKernel(x1, x2, sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \n\t%g\n', sigma, sim);
%===========================================

% Load from ex6data2, use nonlinear decision boundary
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
plotData(X, y);   

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);