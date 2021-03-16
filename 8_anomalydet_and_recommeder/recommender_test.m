% implement the collaborative filtering learning algorithm 
% and apply it to a dataset of movie ratings of scale 1 to 5
% n_u = 943, n_m = 1682

% Load data
load('ex8_movies.mat');
% From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));

% We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');

% Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
% Expect to see a cost of 22.22 for lambda = 0.
load('ex8_movieParams.mat');

% Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

% Evaluate cost function
J = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, num_movies,num_features, 0);
fprintf('Cost at loaded parameters: %f ',J);

% Check gradients by running checkNNGradients
checkCostFunction;

% Evaluate regularized cost function
% Expect to see a cost of 31.34 for lambda = 1.5.
J = cofiCostFunc([X(:); Theta(:)], Y, R, num_users, num_movies, num_features, 1.5);      
fprintf('Cost at loaded parameters (lambda = 1.5): %f',J);

%  Check regularized gradients by running checkNNGradients
checkCostFunction(1.5);