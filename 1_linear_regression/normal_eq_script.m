% Solve with normal equations:
% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations:\n%f\n%f\n%f', theta(1),theta(2),theta(3));

% Estimate the price of a 1650 sq-ft, 3 br house.
price = [1,1650,3]*theta; 
fprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f', price);  