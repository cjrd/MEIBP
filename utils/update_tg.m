function [model, aux] = update_tg(model, aux, rand_k_order)
% function [model, aux] = update_tg(model, aux, rand_k_order)
%
% update parameters of variational truncated Gaussian
%
% INPUT
% model: MEIBP model object
% aux:  MEIBP aux object
% rand_k_order: boolean indicating whether to update the entries in a random order (default false)
%
% OUTPUT
% model: updated model with updated truncated Gaussian parameters
% aux: updated aux with updated truncated Gaussian parameters
%
% author: Colorado Reed, gmail address: colorado.j.reed

% update variational stds
K = model.K;
denoms = sqrt(aux.mks + (model.sigX/model.sigA)^2);
tmp = denoms(1,:)';
model.VAsig = model.sigX ./ tmp(:, ones(1, model.D));

% prep matrix computations
k_nodiag = true(model.K);
k_nodiag(1:K+1:K^2) = false;
if aux.has_test
    len_deqc = length(aux.d_eqc);
else
    len_deqc = 0;
end
zero_upvals = zeros(1,aux.exa_uplag) + NaN;
upvals = zero_upvals;
exa_ct = 0;

if rand_k_order
   korder = randperm(K);
else
   korder = 1:K;
end

% carry out the updates
for k = korder
    exa_ct = exa_ct + 1;
    notk = k_nodiag(k,:);
    model.VAmu(k,aux.not_dtinds) = ( aux.ZtX(k,aux.not_dtinds)...
        -  reshape(aux.ZtZ(1, k, notk),1, K - 1)*aux.Ex_A(notk,aux.not_dtinds) ) / denoms(1,k).^2; %
    
    for ii = 1:len_deqc
        ds=aux.d_eqc{ii};
        model.VAmu(k,ds) = ( aux.ZtX(k,ds)...
            -  reshape(aux.ZtZ(ii+1, k, notk), 1, K - 1)*aux.Ex_A(notk, ds) ) / denoms(ii+1, k).^2; 
        model.VAsig(k,ds) = model.sigX./denoms(ii+1,k);
    end
    
    upvals(exa_ct) = k;
    
    % erf is slow -- use periodically to increase inference speed
    if exa_ct == aux.exa_uplag
        aux.Ex_A(upvals, :) = model.VAmu(upvals,:) + aux.consts.sqrt2opi .* model.VAsig(upvals, :) ./ erfcx(model.VAmu(upvals,:) ./ (model.VAsig(upvals, :).*-aux.consts.sqrt2));
        upvals = zero_upvals;
        exa_ct = 0;
    end
end

% compute any remaining expectations
if exa_ct > 0
    upvals = upvals(~isnan(upvals));
   aux.Ex_A(upvals, :) = model.VAmu(upvals,:) + aux.consts.sqrt2opi .* model.VAsig(upvals, :) ./ erfcx(model.VAmu(upvals,:) ./ (model.VAsig(upvals, :).*-aux.consts.sqrt2)); 
end

% update the squared expectations
aux.Ex_Asq = model.VAmu.*aux.Ex_A + model.VAsig.^2;

