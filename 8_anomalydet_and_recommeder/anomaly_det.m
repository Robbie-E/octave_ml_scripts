% We detect anomalous behavior in server computers
% features: throughput (mbps) and latency (ms) of response
% Vast majority are normally operating servers

% The following command loads the dataset. You should now have the variables X, Xval, yval in your environment
load('ex8data1.mat');

% Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

%  Estimate mu and sigma2, not concerning on correlations
[mu, sigma2] = estimateGaussian(X);

%  Density of the multivariate normal at each data point (row) of X (train)
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

%  Density of the multivariate normal at each data point (row) of Xval
pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

%=====================================================
% Loads the second dataset. 
% Gives variables X, Xval, yval in your environment
% Each example has 11 features
load('ex8data2.mat');

% Apply the same steps to the larger dataset
[mu, sigma2] = estimateGaussian(X);

% Training set 
p = multivariateGaussian(X, mu, sigma2);

% Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

% Find the best threshold
% Epsilon is about 1.38e-18, and 117 anomalies are found
[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));