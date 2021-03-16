% Image compression using K-means
% Reduce number of colors of an image to 16 (uses only 4 bits; 2^4 = 16)
% Original image is represented using 24-bit colors
% each RGB channel is represented as 8-bit unsigned integer, 0-255 (2^8 = 256)


%  Load an image of a bird
A = double(imread('bird_small.png'));
A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16;
max_iters = 10;

% Randomly initialize K centroids from the training data
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, ~] = runkMeans(X, initial_centroids, max_iters);

% Find closest cluster members
% map each pixel (specified by index in idx) to the centroid value
idx = findClosestCentroids(X, centroids);
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
figure;
subplot(1, 2, 1);
imagesc(A); 
title('Original');
axis square

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
axis square