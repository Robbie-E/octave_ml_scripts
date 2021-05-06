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
## @deftypefn {} {@var{retval} =} normalizeRatings (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-11

function [Ynorm, Ymean] = normalizeRatings(Y, R)
%NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
%movie (every row)
%   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
%   has a rating of 0 on average, and returns the mean rating in Ymean.

[m, n] = size(Y);
% Non-vectorized implementation
##Ymean = zeros(m, 1);
##Ynorm = zeros(size(Y));
##for i = 1:m
##    idx = find(R(i, :) == 1);
##    Ymean(i) = mean(Y(i, idx));
##    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
##end

% Vectorized implementation
num_user_movie_ratings = sum(R,2)
Ymean = sum(Y.*R,2)./num_user_movie_ratings;
Ynorm = Y - kron(ones(1,n),Ymean);
endfunction
