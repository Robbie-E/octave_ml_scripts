% Run PCA on face images to perform dimension reduction

%  Load Face dataset, 32 x 32 px grayscale images
% There are n = 1024 features 
load ('ex7faces.mat')
%  Display the first 100 faces in the dataset
close all;
displayData(X(1:100, :));

% Normalize X
[X_norm, ~, ~] = featureNormalize(X);

% Run PCA
[U, ~] = pca(X_norm);

% Visualize the top 36 eigenvectors found
displayData(U(:, 1:36)');

% Project face dataset onto the first 100 principal components
% Each image is a vector z(i) in R^100
K = 100;
Z = projectData(X_norm, U, K);

fprintf('The projected data Z has a size of: %d x %d', size(Z));

% Recover data to see what information is lost in dim red
% This can be used to speed up facial recognition, e.g. using neural nets
X_rec  = recoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
displayData(X_norm(1:100,:));
title('Original faces');
axis square;

% Display reconstructed data from only K eigenfaces
subplot(1, 2, 2);
displayData(X_rec(1:100,:));
title('Recovered faces');
axis square;