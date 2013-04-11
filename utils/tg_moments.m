function [exp_A, exp_Asq] = tg_moments(VAmu, VAsig)
% function [exp_A, exp_Asq] = tg_moments(VAmu, VAsig)
% 
% Computes the first two moments of a truncated Gaussian
%
% INPUT
% VAmu: mean of truncated Gaussian
% VASig: std of truncated Gaussian
%
% OUTPUT
% exp_A: Expected value of truncated Gaussian
% exp_Asq: Squared expected value of truncated Gaussian
%
% author: Colorado Reed, gmail address: colorado.j.reed

scale = sqrt(2/pi) .* VAsig ./ erfcx(VAmu ./ (VAsig.*-sqrt(2)));
exp_A = VAmu + scale;
exp_Asq = VAmu.^2 + VAsig.^2 + VAmu .* scale;


