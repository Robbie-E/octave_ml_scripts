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
## @deftypefn {} {@var{retval} =} linearRegCostFunction (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-04

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad
%   X is a design matrix

% Initialize some useful values
m = length(y); % number of training examples
predictions = X*theta;
deviations = predictions - y;
theta_regul = [0; theta(2:end)];

%regCostTerm = (lambda/(2*m))*sum(theta_regul.*theta_regul, 'all');

regCostTerm = (lambda/(2*m))*sum(theta_regul.*theta_regul,1);
J = (1/(2*m))*(deviations)'*deviations + regCostTerm;
regGradTerm = (lambda/m)*theta_regul;
grad = (1/m)*X'*deviations + regGradTerm;
grad = grad(:);
endfunction
