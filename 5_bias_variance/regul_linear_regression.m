% Implement regularized linear regression to predict 
% the amount of water flowing out of a dam using 
% the change of water level in a reservoir.
% A training set that your model will learn on: X, y
% A cross validation set for determining the lambda: Xval, yval
% A test set with unseen examples for evaluating performance

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
% m = Number of examples
m = size(X, 1);

##% Plot training data
##figure;
##plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
##xlabel('Change in water level (x)');
##ylabel('Water flowing out of the dam (y)');

%===================================================================
% Test cost function, theta at [1; 1], with cost 303.993
theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf('Cost at theta = [1 ; 1]: %f', J);

% Test gradient, theta at [1; 1], with gradient [-15.30; 598.250]
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);
fprintf('Gradient at theta = [1 ; 1]:  [%f; %f] \n',grad(1), grad(2));
%======================================================================

##%  Train linear regression with lambda = 0
##lambda = 0;
##[theta] = trainLinearReg([ones(m, 1) X], y, lambda);
##
##%  Plot fit over the data
##figure;
##plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
##xlabel('Change in water level (x)');
##ylabel('Water flowing out of the dam (y)');
##hold on;
##plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
##hold off;
##
##% Generate learning curves
##lambda = 0;
##[error_train, error_val] = learningCurve([ones(m, 1) X], y, [ones(size(Xval, 1), 1) Xval], yval, lambda);
##
##% Plot the training and cross-validation error
##plot(1:m, error_train, 1:m, error_val);
##title('Learning curve for linear regression')
##legend('Train', 'Cross Validation')
##xlabel('Number of training examples')
##ylabel('Error')
##axis([0 13 0 150])
##
##% Print data points
##fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
##for i = 1:m
##    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
##end

%==================================================================
% Polynomial regression and feature normalization
p = 8; %degree of polynomial

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = X_poly_val-mu; % uses implicit expansion instead of bsxfun
X_poly_val = X_poly_val./sigma; % uses implicit expansion instead of bsxfun
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = X_poly_test-mu; % uses implicit expansion instead of bsxfun
X_poly_test = X_poly_test./sigma; % uses implicit expansion instead of bsxfun
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));
%====================================================================

##% Train the model
##lambda = 0;
##[theta] = trainLinearReg(X_poly, y, lambda);
##
##% Plot training data and fit
##plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
##plotFit(min(X), max(X), mu, sigma, theta, p);
##xlabel('Change in water level (x)');
##ylabel('Water flowing out of the dam (y)');
##title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
##
##% Plot learning curve
##[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
##plot(1:m, error_train, 1:m, error_val);
##title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
##xlabel('Number of training examples')
##ylabel('Error')
##axis([0 13 0 100])
##legend('Train', 'Cross Validation')

%=====================================================================
% Adjusting regularization parameter
% Choose the value of lambda
lambda = 1;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

% Generate learning curve
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')
%=============================================================

% Selecting lambda from cross-validation set
% Generate validation curve
[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval);
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

% Print the errors as function of lambda
for i = 1:length(lambda_vec)
    if i == 1
        fprintf('lambda\t\tTrain Error\tValidation Error\n');
    end
    fprintf('%f\t%f\t%f\n',lambda_vec(i), error_train(i), error_val(i));
end

% Computing test error using the best lambda from cross-validation
lambda = 3;
[theta] = trainLinearReg(X_poly_test, ytest, lambda);
[error_test, ~] = linearRegCostFunction(X_poly_test, ytest, theta, 0)