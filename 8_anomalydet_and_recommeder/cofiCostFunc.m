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
## @deftypefn {} {@var{retval} =} cofiCostFunc (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-11

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.

% Unfold the X and Theta matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta

H = X*Theta';
sq_dev = (H - Y).^2;
regCost_term = 0.5*lambda*(sum(sum(X.*X)) + sum(sum(Theta.*Theta)));
J = 0.5*sum(sum(sq_dev.*R)) + regCost_term;

% each row contains gradient row vectors for each movie
regX_grad_term = lambda*X;
X_grad = ((H - Y).*R)*Theta + regX_grad_term;

% each row contains gradient row vectors for each user
regTheta_grad_term = lambda*Theta;
Theta_grad = ((H - Y).*R)'*X + regTheta_grad_term; 

grad = [X_grad(:); Theta_grad(:)];
endfunction
