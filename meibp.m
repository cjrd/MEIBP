function rr_res=meibp(X, params)
% function rr_res=meibp(X, params)
% 
% Method performs MEIBP inference for the matrix factorization model: X = Z*A + e, 
% where Z ~ IBP, A ~ TruncatedGaussian, and e is Gaussian white noise.
% C. Reed & Z. Ghahramani (2013)
% Scaling the Indian Buffet Process via Submodular Maximization
% http://www.arxiv.org/abs/1304.3285
%
% X is the observed data matrix with N instances with D dimensions
% params is a struct with the following options (see meibp_test.m for example usage)
%     'Kinit' : initial number of latent features
%     'test_mask',  sparse NxD matrix where a entry n,d=1 indicates observation n,d should be masked
%     'num_restarts' : number of random restarts
%     'Zinit_per' : fraction of Z entries that should initially equal 1 for random initialization
%     'alpha', IBP hyperparameter (number of latent features ~ alpha*log(N))
%     'sigX', Gaussian Hyperparameter (model noise)
%     'sigA', Gaussian Hyperparameter (latent feature noise)
%     'use_runtime', whether to use an upperbound on inference time
%     'max_runtime', the upperbound on inference time
%     'chk_conv', whether to check for model convergence
%     'conv_method', check for convergence using the variational lower bound 'vlb' or training log likelihood 'tll'
%     'conv_thresh', relative difference in conv_method that indicates convergence
%     'conv_window', use average convergence criterion in an (conv_window)-block window
%     'exa_uplag', how often to update the A expectations (every n A_i feature updates); generally higher values decrease runtime and results
%     'detailed_vlb',  % keep track of all vlb changes (much slower but good for debugging)
%     'exch_ec', % specify true to use the lof IBP prior or false to use shifted equivalence class prior
%     'rand_n_order', set 'true' to update the instances randomly or false to update the instances sequentially
%     'update_a_each_zn', 'true' updates the A variational distributiona after each Z_n optimization (slower but performs much better in practice)
%     );
%
% author: Colorado Reed, gmail address: colorado.j.reed

fprintf('!!!! Beginning MEIBP for N=%i, Kinit = %i !!!!\n', size(X,1), params.Kinit);

rr_res(params.num_restarts) = struct('model' ,[], 'aux', [], 'vlb', []); % preallocation

for rr = 1:params.num_restarts
    %-----------------------%
    %    initialization     %
    %-----------------------%
    clear aux model update_mat
    model.K = params.Kinit;
    model.N = size(X, 1);
    model.D = size(X, 2);
    Xtotal = X;
    aux.test_mask = params.test_mask;
    aux.NDtrain = model.D*model.N - sum(aux.test_mask(:));
    aux.has_test = model.D*model.N > aux.NDtrain;
    if aux.has_test
        X(aux.test_mask) = 0;
    end
    
    fprintf('\n------ beginning restart %i of %i ------\n', rr,params.num_restarts)
    eval_res.l2 = zeros(500,1) + NaN;
    eval_res.ll = zeros(500,1) + NaN;
    eval_res.ll_train = zeros(500,1) + NaN;
    
    % set the model and aux params (order matters)
    model.alpha = params.alpha;
    model.sigX = params.sigX;
    model.sigA = params.sigA;
    model.Z = rand(model.N, model.K) < params.Zinit_per;
    model.VAmu = abs(0.05 * randn(model.K, model.D));
    VAsig_tmp = abs(randn(model.K, 1)*0.1)  ;
    model.VAsig = VAsig_tmp(:, ones(1, model.D)); % TODO only store the vector?
    [aux.Ex_A, aux.Ex_Asq] = tg_moments(model.VAmu, model.VAsig);
    aux.trXX = sum(sum(X.^2));
    EULERGAMMA = 0.5772156649015328606;
    aux.Hn = EULERGAMMA + psi(model.N+1); % compute the nth harmonic number
    aux.vlb_ct = 1;
    aux.all_vlb = params.detailed_vlb;
    aux.vlb_itend = [];
    aux.exch_ec = params.exch_ec;
    aux.exa_uplag = params.exa_uplag;
    % small time savers
    aux.gammaln_mat = gammaln(1:max(model.N,model.K)+1); % more efficient than repeated computations (even for massive N, N=10^7)
    aux.consts.sqrt2 = sqrt(2);
    aux.consts.sqrt2opi = sqrt(2/pi);
    
    aux.mks(1,:) = sum(model.Z,1);
    % compute first term (case: no d missing) and the rest (ec for missing d)
    aux.ZtX = model.Z'*X; % NKD
    aux.ZtZ(1,:,:) = double(model.Z)'*double(model.Z); % NK^2 (D)
    
    %-----------------------%
    %    Handle Test Mask   %
    %-----------------------%
    %%% initialize rank 1 feature update matrices for test data %%%
    aux.testn = sum(aux.test_mask,2) > 0;
    dinds = find(sum(aux.test_mask,1));
    aux.not_dtinds = setdiff(1:model.D, dinds);
    if aux.has_test
        % construct equivalent classes with test dimensions
        [~, c_types, eq_inds]=unique(aux.test_mask(:,dinds)', 'rows');
        aux.d_eqc{length(c_types)} = [];
        for i=1:length(c_types)
            aux.d_eqc{i} =  dinds(eq_inds==i);
        end
        
        % find mask values
        mask_by_d{length(aux.d_eqc)} = [];
        for i=1:length(aux.d_eqc)
            mask_by_d{i} = find(aux.test_mask(:,aux.d_eqc{i}(1)));
        end
        
        for i=1:length(aux.d_eqc)
            useZ = model.Z;
            useZ(mask_by_d{i},:) = 0;
            aux.ZtZ(i+1,:,:) = double(useZ)'*double(useZ);
            aux.mks(i+1,:) = sum(useZ,1);
        end
    end
    
    if aux.all_vlb
        aux.vlb_zind = [];
        aux.vlb_aind = [];
    end
    
    % initialize vlb and model
    aux.vlb(aux.vlb_ct) = compute_vlb(X, model, aux);
    aux.vlb_ct = aux.vlb_ct +1;
    [model, aux] = update_tg(model, aux, params.rand_k_order);

    %-----------------------%
    %    MEIBP Iterations   %
    %-----------------------%
    times = zeros(200,1) + NaN;
    converge = false;
    ii =1;
    tot_runtime = 0;
    stime = tic;
    while ~converge 
        fprintf('iteration %d \n',ii);

        if params.rand_n_order
            nperm = randperm(model.N);
        else
            nperm = 1:model.N;
        end
        
        for n=nperm
            %-----------------------%
            %    Optimization       %
            %-----------------------%
            % comp_res(end+1,:)
            newZn = local_search_opt(model, aux, X(n,:), n);
            
            % update A if Zn changed 
            if (~isequal(newZn, model.Z(n,:)) && params.update_a_each_zn) || (~params.update_a_each_zn && n==nperm(end))
                zn_diff = newZn - model.Z(n,:);
                aux.mks(1,:) = aux.mks(1,:) + zn_diff;
                new_k2 = double(newZn)'*double(newZn);  %  Z should be sparse
                old_k2 = double(model.Z(n,:))'*double(model.Z(n,:));  %  Z should be sparse
                model.Z(n,:) = newZn;
                
                % update ZtZ
                update_mat(1,:,:) = new_k2 - old_k2;
                aux.ZtZ(1,:,:) = aux.ZtZ(1,:,:) + update_mat(1,:,:);
                if aux.has_test
                    for i=1:length(aux.d_eqc)
                        if ~aux.test_mask(n,aux.d_eqc{i}(1))% don't update if nd is masked
                            aux.ZtZ(i+1,:,:) = aux.ZtZ(i+1,:,:) + update_mat;
                            aux.mks(i+1,:) = aux.mks(i+1,:) + zn_diff;
                        end
                    end
                end
                
                % update ZtX
                aux.ZtX = aux.ZtX + zn_diff'*X(n,:);
                zero_idxs = find(aux.mks(1,:)==0);
                
                % remove empty features
                if ~isempty(zero_idxs)
                    clear update_mat;
                    [model, aux] = remove_features(model, aux, zero_idxs);
                end
                
                if aux.all_vlb
                    aux.vlb(aux.vlb_ct) = compute_vlb(X, model, aux);
                    aux.vlb_zind(end+1) = aux.vlb_ct;
                    aux.vlb_ct = aux.vlb_ct +1;
                end
                
                % optimize latent feature variational dists
                [model, aux] = update_tg(model, aux, params.rand_k_order);
            end
            % possible with overspecified models
            if model.K == 0
                break
            end
        end
        times(ii) = toc(stime); 
        tot_runtime = tot_runtime + times(ii);
        
        %%%%%%%%%%%%%%%%%
        %   ERROR EVAL  %
        %%%%%%%%%%%%%%%%%
        eval_res.l2(ii) = ibp_error_eval(Xtotal, model.Z*aux.Ex_A, aux.test_mask);
        [eval_res.ll_train(ii),eval_res.ll(ii)] =  uncoll_llhood(model.Z , Xtotal, model.sigX , aux.Ex_A , [] , params);
        
        % record vlb, check for convergence, print updates
        aux.vlb(aux.vlb_ct) = compute_vlb(X, model, aux);
        aux.vlb_itend(end+1) = aux.vlb_ct;
        aux.vlb_ct = aux.vlb_ct +1;
        if  ii > 2*params.conv_window
            if strcmp(params.conv_method, 'vlb')
                conv_vals = aux.vlb(1:aux.vlb_ct-1);
            else % use training likelihood
                conv_vals = eval_res.ll_train;
            end
            
            useinds = (ii-params.conv_window+1):ii;
            cur_mean = log_mean(conv_vals, useinds);
            useinds = (ii-2*params.conv_window+1):(ii-params.conv_window);
            prev_mean = log_mean(conv_vals, useinds);
            conv_meas =  abs((cur_mean-prev_mean)/(prev_mean));

            if params.chk_conv && conv_meas <= params.conv_thresh 
                converge = true;
            end
        else
            conv_meas = -1;
        end
        
        % print console updates
        fprintf('K: %-3i vlb: %6.3f conv: %0.5f itr time %0.3f \n', model.K, aux.vlb(end), conv_meas, times(ii));
        
        ii = ii + 1;
        if params.use_runtime
            if params.max_runtime < tot_runtime
                break;
            end
        end
        
        if model.K == 0
            break
        end
        
        stime=tic;
    end
    fprintf('\ntotal inference time: %0.2fs for N=%i \n\n', tot_runtime, model.N);
    eval_res.l2 = eval_res.l2(~isnan(eval_res.l2));
    eval_res.ll = eval_res.ll(~isnan(eval_res.ll));
    aux.eval_res = eval_res;
    aux.times = times(~isnan(times));
    aux.final_its = ii -1;
    
    % save the model
    rr_res(rr) = struct('model' ,model, 'aux', aux, 'vlb', aux.vlb(aux.vlb_ct-1));
end
end

function [model, aux] = remove_features(model, aux, zero_idxs)
% remove features from model and aux
model.Z(:,zero_idxs) = [];
model.VAmu(zero_idxs, :) = [];
model.VAsig(zero_idxs, :) = [];
model.K = size(model.Z, 2);
aux.Ex_A(zero_idxs, :) = [];
aux.Ex_Asq(zero_idxs, :) = [];
aux.mks(:, zero_idxs) = [];
aux.ZtZ(:, zero_idxs, :) = [];
aux.ZtZ(:, :, zero_idxs) = [];
aux.ZtX(zero_idxs, :) = [];
end
