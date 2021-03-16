% implement the collaborative filtering learning algorithm 
% and apply it to a dataset of movie ratings of scale 1 to 5
% n_u = 943, n_m = 1682, n = 10
% Goal is to predict score to movies you haven't watched using Theta and X
% You can also determine what movies are close to each other from learned X

% Load movie list
movieList = loadMovieList();

% Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;
% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2; 

% We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

##fprintf('\n\nNew user ratings:\n');
##for i = 1:length(my_ratings)
##    if my_ratings(i) > 0 
##        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
##    end
##end

% Load data containing Y and R
load('ex8_movies.mat');
% Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
% R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
% Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

% Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

% Set the number of features n=10
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj','on','MaxIter',100);

% Set Regularization
lambda = 10;
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), initial_parameters, options);

% Unfold the returned theta back into X and Theta
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

% Predict ratings, each column is for a user
p = X*Theta';
my_predictions = p(:,1) + Ymean;

% Sort predicted ratings for one user, highest to lowest
% get top 10 recommended movies for you
[r, ix] = sort(my_predictions,'descend');
for ranking = 1:10
    % get movie index i
    i = ix(ranking);
    if ranking == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(i), movieList{i});
end

% Original ratings provided by the user
for i = 1:length(my_ratings)
    if i == 1
        fprintf('\n\nOriginal ratings provided:\n');
    end
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end