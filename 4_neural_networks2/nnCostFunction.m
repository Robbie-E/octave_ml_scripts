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
## @deftypefn {} {@var{retval} =} nnCostFunction (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-02

function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) X]; %add bias to a1, m x (s1 + 1)
z2 = (Theta1*a1')'; %m x s2
a2 = [ones(m, 1) sigmoid(z2)]; %add bias to a2, m x (s2 + 1)
z3 = (Theta2*a2')'; %m x s3
a3 = sigmoid(z3);
% output max probabilities and corresponding class labels
% each sample is a row
%[max_probabilities , predicted_labels] = max(a3, [], 2);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

y_mapped = zeros(m, num_labels);
for i = 1:m
    y_mapped(i,y(i)) = 1;
end

delta3 = a3' - y_mapped'; %(s3 x m)
delta2 = Theta2'*delta3.*(a2.*(1-a2))'; %(s2 x m)

% Part 3: Implement regularization with the cost function and gradients.
% Cost from forward propagation
Theta1_sq = Theta1(:,2:end).*Theta1(:,2:end);
Theta2_sq = Theta2(:,2:end).*Theta2(:,2:end);
reg_costterm1 = (lambda/(2*m))*sum(Theta1_sq(:));
reg_costterm2 = (lambda/(2*m))*sum(Theta2_sq(:));

% for matlab, this is shortened
%reg_costterm1 = (lambda/(2*m))*sum(Theta1(:,2:end).*Theta1(:,2:end), 'all');
%reg_costterm2 = (lambda/(2*m))*sum(Theta2(:,2:end).*Theta2(:,2:end), 'all');

J = (-1/m)*trace(y_mapped*log(a3')) + (-1/m)*trace((1-y_mapped)*log((1-a3)')) + reg_costterm1 + reg_costterm2;

% Gradients from backpropagation
reg_gradterm1 = (lambda/m)*[zeros(hidden_layer_size,1) Theta1(:,2:end)]; %theta1,0 not regularized
reg_gradterm2 = (lambda/m)*[zeros(num_labels,1) Theta2(:,2:end)]; %theta2,0 not regularized
grad_theta1_term = delta2*a1; %remove first row (remove bias in layer 2)
grad_Theta1 = (1/m)*grad_theta1_term(2:end,:) + reg_gradterm1; 
grad_Theta2 = (1/m)*delta3*a2 + reg_gradterm2;

% Unroll gradients
grad = [grad_Theta1(:) ; grad_Theta2(:)];
endfunction
