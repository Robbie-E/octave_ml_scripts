% K-means used in 3-D RGB space to compress an image to K=16 colors
% Visualize the final pixel assignments in this 3D space
% Each pixel point is colored according to cluster assignment

A = double(imread('bird_small.png'));
% Normalize pixel values (0-255 to 0-1)
A = A / 255;
img_size = size(A);
% we have 3 dimensions for RGB channels
X = reshape(A, img_size(1) * img_size(2), 3);

% Run K-means on the image
K = 16; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
sel = floor(rand(1000, 1) * size(X, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');

% PCA projection as a rotation that selects the view 
% that maximizes the spread of the data.
% Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);

% PCA and project the data to 2D
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

% Plot in 2D
figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');