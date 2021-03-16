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
## @deftypefn {} {@var{retval} =} lrCostFunction (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-01

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.
%   X is design matrix  

% Initialize some useful values
m = length(y); % number of training examples

predictions = sigmoid(X*theta);
reg_costterm = (lambda/(2*m))*((theta'*theta)-(theta(1)^2));
J = (-1/m)*(y'*log(predictions))+(-1/m)*((1-y)'*log(1-predictions)) + reg_costterm;
reg_gradterm = (lambda/m)*[0;theta(2:length(theta))]; %theta0 not regularized
grad = (1/m)*X'*(predictions-y) + reg_gradterm;
endfunction
