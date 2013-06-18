function testvec = local_search_opt(model, aux, Xn, n)
% function testvec = local_search_opt(model, aux, Xn, n)
% 
% performs submodular local search optimization from "Maximizing non-monotone submodular functions" (Feige et al 2007)
% for the MEIBP model descibed in
% C. Reed & Z. Ghahramani (2013)
% Scaling the Indian Buffet Process via Submodular Maximization
% http://www.arxiv.org/abs/1304.3285
%
%
% INPUT
% model: MEIBP inference model object
% aux: MEIBP inference aux object
% Xn: the D-dimensional nth row of data matrix X
% n: integer representing the nth row use for Xn (used to determine test mask)
% 
% OUTPUT
% testvec: the latent optimized feature assignment for Zn
% 
% author: Colorado Reed, gmail address: colorado.j.reed

%%% construct quadratic term
if aux.testn(n)
    test_mask_n = aux.test_mask(n,:);
    rmat = test_mask_n(ones(model.K,1),:);
    exA_tilde = aux.Ex_A;
    exA_tilde(rmat) = 0;
    exAsq_tilde = aux.Ex_Asq;
    exAsq_tilde(rmat) = 0;
    sum_trm = -1/2*(sum(exAsq_tilde,2));% - sum(aux.Ex_Asq(rmat),2));
    W = -1/model.sigX^2*(exA_tilde*exA_tilde');
else
    sum_trm = 1/2* (-sum(aux.Ex_Asq, 2) );
    W = -1/model.sigX^2*(aux.Ex_A*aux.Ex_A');
end

%%% construct linear term
omegan = Xn*aux.Ex_A'; 
omegan = (omegan + sum_trm')./model.sigX^2;
mks_zn = aux.mks(1,:) - model.Z(n,:);
kvals =1:model.K;
nutrm = aux.gammaln_mat(model.N-mks_zn) + aux.gammaln_mat(mks_zn + 1); % znk=1 case
nutrm(mks_zn > 0) = nutrm(mks_zn > 0) - ( aux.gammaln_mat(model.N-mks_zn(mks_zn > 0) + 1) + aux.gammaln_mat(mks_zn(mks_zn > 0)) );  %znk=0 case
ckc_trm = zeros(1,model.K);
if ~isempty(find(~mks_zn, 1))
    rvals = ~mks_zn;
    ckc_trm(rvals) = log(model.alpha) - aux.gammaln_mat(model.N+1) - model.D/2*log(pi*model.sigA^2/2);
    for rk=find(rvals) 
        ckc_trm(rk) = ckc_trm(rk) + sum(-1*aux.Ex_Asq(rk,:)./(2*model.sigA^2)...
            + comp_tg_entropy(model.VAmu(rk,:), model.VAsig(rk,:), aux.Ex_A(rk,:), aux.Ex_Asq(rk,:))); % ck term
    end
end
omegan = omegan + nutrm + ckc_trm;

%%% perform optimization
testvec = zeros(1,model.K);
[initmax,initk] = max(omegan);
if initmax < 0
   return 
end
tot_diff = initmax;
testvec(initk) = 1;
combtots = omegan + W(initk,:);
% initializtion
TOL = 0; % TODO make this an option
curset = initk;
unadd_vals = kvals; %sfo_setdiff_fast(inset, curset);
unadd_vals(initk) = [];
i = 0;
wdiag = diag(W)';
while true
    i = i+1;
    for up_pass=[true false]
        down_change = false;
        if up_pass
            while true
                if isempty(unadd_vals)
                    break
                end
                [maxval, maxind] = max(combtots(unadd_vals));
                if maxval < TOL
                    break
                end
                maxk = unadd_vals(maxind); 
                curset = [curset maxk];
                testvec(maxk) = 1;
                combtots = combtots + W(maxk,:);
                tot_diff = tot_diff + maxval;
                unadd_vals(maxind) = [];
            end
        else
            while true 
                [maxval, maxind] = min(combtots(curset) - wdiag(curset));  
                if -maxval < TOL
                    break
                end
                down_change = true;
                maxk = curset(maxind); 
                curset(maxind) = [];
                unadd_vals = [unadd_vals maxk];
                testvec(maxk) = 0;
                combtots = combtots - W(maxk,:);
                tot_diff = tot_diff - maxval;
                break % can remove this and still have a valid algorithm
            end
        end
    end % end up/down pass loop
    
    % check for convergence
    if tot_diff <= TOL || ~down_change
        break;
    end
    tot_diff = 0;
end % end outter while

end




