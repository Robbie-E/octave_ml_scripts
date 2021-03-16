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
## @deftypefn {} {@var{retval} =} dataset3Params (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-05

function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set. Return optimal C and sigma
%   X, Xval are not design matrices

C_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
val_errors = zeros(length(C_vec), length(sigma_vec));

for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        model = svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j))); 
        val_pred = svmPredict(model, Xval);
        % Calculate misclassification error
        val_errors(i,j) = mean(double(val_pred ~= yval));
    end
end

min_val_error = min(min(val_errors));
[C_min_i, sigma_min_j] = find(val_errors== min_val_error);
C = C_vec(C_min_i);
sigma = sigma_vec(sigma_min_j);
endfunction
