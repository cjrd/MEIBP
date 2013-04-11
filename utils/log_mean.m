function res = log_mean(vals, useinds)
% function res = log_mean(vals, useinds)
%
% Computes the log of the mean of a set of input log values
%
% INPUT
% vals: the input log values
% useinds: the indices of vals to use in the mean (default: 1:length(vals))
%
% author: Colorado Reed, gmail address: colorado.j.reed

if ~exist('useinds','var')
    useinds=1:length(vals);
end                      
offset = -1 * ( max( vals(useinds) ) - 2 );   
res = log( sum( exp( vals(useinds) + offset ) ) ) - log(length(useinds)) - offset;
