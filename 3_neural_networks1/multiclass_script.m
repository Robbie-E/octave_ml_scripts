% recognize hand-written digits
% There are 5000 training examples in ex3data1.mat, 
% each training example is a 20 pixel by 20 pixel digit grayscale image
% Each pixel is represented by a floating point number, 
% the grayscale intensity at that location. 
% The 20 by 20 grid of pixels is 'unrolled' into a 400-dimensional vector.
% Map digits to classes k = 1...10 (10 is for '0' digit)

% Load saved matrices from file
load('ex3data1.mat');
% The matrices X and y will now be in your MATLAB environment
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

%test cost function results
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('Cost: %f | Expected cost: 2.534819\n',J);
fprintf('Gradients:\n'); fprintf('%f\n',grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003');

%One-vs-all classification
num_labels = 10; % 10 labels, from 1 to 10 
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

% predict from training set (94.9 % accuracy)
pred = predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
