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
## @deftypefn {} {@var{retval} =} oneVsAll (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-01

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the k-th row of all_theta corresponds 
%   to the classifier for label k

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n+1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Note: We recommend using fmincg to optimize the cost
%       function. fmincg works similarly to fminunc, 
%       but is more efficient when we are dealing
%       with large number of parameters.

for k = 1:num_labels
    [all_theta(k,:)] = fmincg(@(t)(lrCostFunction(t,X,(y==k),lambda)), all_theta(k,:)', options)';
end
endfunction
