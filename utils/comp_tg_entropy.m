function res = comp_tg_entropy(VAmu, VAsig, ExA, ExAsq)     
% function res = comp_tg_entropy(VAmu, VAsig, ExA, ExAsq)     
%
% calculate the entropy of a trucated Gaussian
%
% INPUT
% VAmu: mean matrix of truncated Gaussian
% VAsig: std matrix of truncated Gaussian
% ExA: Expected value matrix of truncated Gaussian
% ExAsq: Expected squared value matrix of truncated Gaussian
%
% OUTPUT
% res: truncated Gaussian entropy
% author: Colorado Reed, gmail address: colorado.j.reed
res = -1./(2*VAsig.^2) .* (ExAsq - ExA.^2 + (ExA - VAmu).^2)...
    + 0.5*log(2./(pi.*VAsig.^2)) - log_erfc(-1*(VAmu./(VAsig.*sqrt(2))));
res = res*-1;       