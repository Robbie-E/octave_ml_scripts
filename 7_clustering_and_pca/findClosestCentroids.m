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
## @deftypefn {} {@var{retval} =} findClosestCentroids (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-03-08

function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry idx(i) in range [1..K])
%   idx(i) = should contain the index of the centroid closest to example i

% Set K
K = size(centroids, 1);
m = size(X,1);
idx = zeros(size(X,1), 1);
for example = 1:m
    % dev_from_mean(k) = (x(i) - mu(k))'; Kxn
    dev_from_mean = kron(ones(K,1), X(example,:))-centroids;
    % sq_distances(k,:) = square distance between x(i) and mu(k)
    sq_distances = diag(dev_from_mean*dev_from_mean');
    % column-wise minimum (minimum over rows of sq_distances, over different centroids)
    [~,idx(example)] = min(sq_distances, [], 1);
end
endfunction