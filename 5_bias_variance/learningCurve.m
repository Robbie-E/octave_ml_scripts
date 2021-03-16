## Copyright (C) 2021 Robertson
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} learningCurve (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-05

function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%       X, Xval is a design matrix.
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.

% Naive calculation
%for i = 1:m
%    [theta] = trainLinearReg(X(1:i, :), y(1:i), lambda);
%    [error_train(i), ~] = linearRegCostFunction(X(1:i, :), y(1:i), theta, 0);
%    [error_val(i), ~] = linearRegCostFunction(Xval, yval, theta, 0);
%end

%======================================================================================
% For small training sets, it is often helpful to average across multiple sets
% of randomly selected examples to determine the training error and cross validation error.
% randomly select i examples from the training set and i examples from the cross validation set

num_trials = 50;
% Vary size of training set
for i = 1:m
    % Perform num_trials trials for each training set size i
    for trial = 1:num_trials
        % select i unique integers randomly from 1 to m
        trial_rows_train = randperm(m,i);
        trial_rows_cv = randperm(m,i);
        [theta] = trainLinearReg(X(trial_rows_train, :), y(trial_rows_train), lambda);
        [error_train_trial, ~] = linearRegCostFunction(X(trial_rows_train, :), y(trial_rows_train), theta, 0);
        [error_val_trial, ~] = linearRegCostFunction(Xval(trial_rows_cv, :), yval(trial_rows_cv), theta, 0);
        error_train(i) = error_train(i) + error_train_trial;
        error_val(i) = error_val(i) + error_val_trial;
    end
end

error_train = error_train./double(num_trials);
error_val   = error_val./double(num_trials);
endfunction
