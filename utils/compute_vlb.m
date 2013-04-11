function [vlb, vlb_llh, vlb_ibp, vlb_lf, vlb_ent]  = compute_vlb(X, model, aux)
% function [vlb, vlb_llh, vlb_ibp, vlb_lf, vlb_ent]  = compute_vlb(X, model, aux)
%
% compute the variational lower bound for the truncated Gaussian MEIBP model
%  
% INPUT
% X : NxD input data matrix
% model : MEIBP model object
% aus: MEIBP aux object
%
% OUTPUT
% vlb: variational lower bound
% vlb_llh: log-likelihood component of variational lower bound 
% vlb_ibp: ibp prior component of variational lower bound 
% vlb_lf: latent factor prior component of variational lower bound 
% vlb_ent: latent factor entropy component of variational lower bound 
%
%
% author: Colorado Reed, gmail address: colorado.j.reed

N = model.N;
D = model.D;
Kplus_idx = find(aux.mks(1,:));
Kplus = length(Kplus_idx);

%%% Likelihood term %%%
vlb_llh = compute_vllh(X, model, aux);

%%% IBP term %%%
n_mks = N - aux.mks(1,:);
if aux.exch_ec
    [~, ~, idx] = unique(nonzeros(check_histories(model.Z(:, Kplus_idx))));
    khtrm = 0;
    if ~isempty(idx)
        khcts = accumarray(idx(:), 1, [], @sum);
        khtrm = sum(aux.gammaln_mat(khcts+1));
    end
else
    khtrm = aux.gammaln_mat(Kplus+1);
end
    vlb_ibp = Kplus*log(model.alpha) - Kplus*aux.gammaln_mat(N+1) - model.alpha*aux.Hn ...
        + sum(aux.gammaln_mat(n_mks(Kplus_idx)+1) + aux.gammaln_mat(aux.mks(1,Kplus_idx))) - khtrm;

%%% Latent feature term %%%
vlb_lf = - Kplus*D/2*log(pi * model.sigA^2/2) - 1/(2*model.sigA^2)*sum(sum(aux.Ex_Asq));

%%% Latent feature entropy term %%%
envals = comp_tg_entropy(model.VAmu, model.VAsig, aux.Ex_A, aux.Ex_Asq);
vlb_ent =  sum(sum(envals));

% total vlb
vlb = vlb_llh + vlb_ibp + vlb_lf + vlb_ent;
