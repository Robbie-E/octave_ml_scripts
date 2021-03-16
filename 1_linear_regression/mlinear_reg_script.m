% You want to know what a good market price would be. 
% You have housing prices in Portland, Oregon. 
% The first column is the size of the house (in square feet), 
% the second column is the number of bedrooms, 
% and the third column is the price of the house.

% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
% First 10 examples from the dataset
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);
%add bias, create design matrix
% Add intercept term to X
X = [ones(m, 1) X];

% Run gradient descent (many features)
% Choose some alpha value
alpha = 0.01;
num_iters = 10;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, ~] = gradientDescent(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f',theta(1),theta(2),theta(3))
computeCost(X, y, theta)

% Estimate the price of a 1650 sq-ft, 3 br house
normed_samp = ([1650,3]-mu)./sigma;
price = ([1,normed_samp]*theta); 
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f', price)

% Adjusting learning rates
% Run gradient descent: choose some alpha value
alpha = 0.01;
num_iters = 200;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[~, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
