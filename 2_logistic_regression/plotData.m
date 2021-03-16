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
## @deftypefn {} {@var{retval} =} plotData (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Robertson <Robertson@LAPTOP-RCE>
## Created: 2021-02-26

function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. x is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

%               Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.

%give row and column indices of elements satitsfying conditional
pos = find(y==1); neg = find(y == 0); %indicate sample index
plot(x(pos, 1), x(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(x(neg, 1), x(neg, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

hold off;
endfunction
