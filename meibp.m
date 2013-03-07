function [model,aux]=meibp(params, X, ZAtrue)

if exist('ZAtrue','var')
   eval_ZAtrue = true;
else
   eval_ZAtrue = false;
end

%-----------------------%
%    initialization     %
%-----------------------%
% zero the test entries
model.N = size(X, 1);
model.D = size(X, 2);
model.K = params.Kinit;
Xtotal = X;
aux.test_mask = params.test_mask;
aux.NDtrain = model.D*model.N - sum(aux.test_mask(:));
aux.has_test = model.D*model.N > aux.NDtrain;

if aux.has_test
   X(aux.test_mask) = 0; 
end

eval_res.l2 = zeros(500,1) + NaN;
eval_res.ll = zeros(500,1) + NaN;
eval_res.ll_train = zeros(500,1) + NaN;
Xfull = X;
if params.data_blocks
    X = Xfull(1:params.data_block_size, :);
    blk_idx = params.data_block_size + 1;
else
    blk_idx = size(X,1) + 1;
end
model.alpha = params.alpha;
model.sigX = params.sigX;
model.sigA = params.sigA;
model.Z = rand(model.N, model.K) < params.Zinit_per; % TODO are there better/sparse ways to represent Z?
model.VAmu = abs(0.05 * randn(model.K, model.D));
VAsig_tmp = abs(randn(model.K, 1)*0.1);
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
aux.gammaln_mat = gammaln(1:max(model.N,model.K)+1); % more efficient than repeated computations (what about for massive N?)
aux.consts.sqrt2 = sqrt(2);
aux.consts.sqrt2opi = sqrt(2/pi);

aux.mks(1,:) = sum(model.Z,1);
% compute first term (case: no d missing) and the rest (ec for missing d)
aux.ZtX = model.Z'*X; % NKD
aux.ZtZ(1,:,:) = double(model.Z)'*double(model.Z); % NK^2 (D)

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
if params.tempering
    params.sigX_orig = model.sigX;
    params.sigA_orig = model.sigA;
end
comp_res = []; % TODO initialize better

% initialize vlb and model
aux.vlb(aux.vlb_ct) = compute_vlb(X, model, aux);
aux.vlb_ct = aux.vlb_ct +1;
[model, aux] = update_tg(model, aux);

% TODO does update ordering matter here? should we update all at once or
% per line -- would have to recompute ZexA each time... What's correct? what
% works?

%-----------------------%
%    CA Iterations      %
%-----------------------%
times = zeros(200,1) + NaN;
converge = false;
ii =1;
tot_runtime = 0;
stime = tic;
while ~converge && ii <= params.maxits % TODO make sure this is larger than data block size
    fprintf('iteration %d \n',ii);
    
    if params.tempering
        model.sigX = params.sigX_orig * (1 + exp(-ii/5));
        model.sigA = params.sigA_orig * (1 + 0.5 * exp(-ii/5));
    end
    
    % optimize subsets and then update the var latent feature dists HACK/HEURISTIC?
    
    if params.first_pass && blk_idx <= size(Xfull, 1) && ii > 1
        tmp_perm = randperm(numadd);
        tmp_arr = model.N - numadd: model.N;
        nperm = tmp_arr(tmp_perm);
    else
        nperm = 1:model.N;%randperm(model.N); % TODO make option
    end
    
    %    npart = mat2cell(nperm , 1, diff(round(linspace(0,model.N, nsub+1))));
    for n=nperm
        %-----------------------%
        %    Optimization       % 
        %-----------------------%
        % comp_res(end+1,:)
        newZn = ls_lazy_inc(model, aux, X(n,:), n);
        % [newZn, ~] = zn_opt(model, aux, n, X(n,:), params.max_opts);
        if ~isequal(newZn, model.Z(n,:))
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
            [model, aux] = update_tg(model, aux);
        end
    end
    
% % %     if params.try_full_flip % TODO should we disable this while adding data?
% % %         full_idxs = find(aux.mks(1,:)==model.N);
% % %         if ~isempty(full_idxs)
% % %             [model, aux] = remove_features(model, aux, full_idxs);
% % %             %             for idx=full_idxs
% % %             %
% % %             %                 %                       vlb_full = compute_vlb(X, model, aux);
% % %             %                 %                       model.Z(:, idx) = 0;
% % %             %                 %                       vlb_empty = compute_vlb(X, model, aux);
% % %             %                 %                       if vlb_empty >= vlb_full
% % %             %                 %                           [
% % %             %                 %                       else
% % %             %                 %                           model.Z(:, idx) = 1;
% % %             %                 %                       end
% % %             %             end
% % %         end
% % %     end
    
    
    %%% add more data
    if params.data_blocks && blk_idx <= size(Xfull, 1)
        endval = min(size(Xfull, 1), (blk_idx + params.data_block_size));
        X = [X; Xfull(blk_idx:endval, :)];
        aux.trXX = sum(sum(X.^2));
        numadd = endval - blk_idx + 1;
        model.Z = [model.Z; rand(numadd, size(model.Z, 2)) < params.Zinit_per];
        model.N = model.N + numadd;
        blk_idx = blk_idx + params.data_block_size + 1;
    end
    
    if params.add_features % only add if llh increases?
        clear update_mat;
        Xresid = X - model.Z*aux.Ex_A;
        Znew = kmeans(Xresid, 2) - 1;
        model.Z = [model.Z, Znew];
        model.VAmu = [model.VAmu; zeros(1, model.D)];
        model.VAsig = [model.VAsig; zeros(1, model.D)];
        aux.Ex_A = [aux.Ex_A; zeros(1, model.D)];
        aux.Ex_Asq = [aux.Ex_Asq; zeros(1, model.D)];
        model.K = model.K + 1;
        %%%%%%%%%%%%%%%%%
        % Update Params %
        %%%%%%%%%%%%%%%%%
        [model, aux] = update_tg(model, aux);
    end
    times(ii) = toc(stime); % TODO make verbose option (save timing in an aux vector as well)
    tot_runtime = tot_runtime + times(ii);
    
    %%%%%%%%%%%%%%%%%
    %   ERROR EVAL  %
    %%%%%%%%%%%%%%%%%
    if eval_ZAtrue
        eval_res.l2(ii) = ibp_error_eval(Xtotal, model.Z*aux.Ex_A, aux.test_mask, ZAtrue); % TODO: implement
    else
        eval_res.l2(ii) = ibp_error_eval(Xtotal, model.Z*aux.Ex_A, aux.test_mask);
    end
    [eval_res.ll_train(ii),eval_res.ll(ii)] =  uncoll_llhood(model.Z , Xtotal, model.sigX , aux.Ex_A , [] , params, params.trans_set);
    
    % record vlb, check for convergence, print updates
    aux.vlb(aux.vlb_ct) = compute_vlb(X, model, aux);
    aux.vlb_itend(end+1) = aux.vlb_ct;
    aux.vlb_ct = aux.vlb_ct +1;
    if  ii > 2*params.conv_window
         useinds = (ii-params.conv_window+1):ii;
        cur_mean = log_mean(eval_res.ll_train, useinds);
        useinds = (ii-2*params.conv_window+1):(ii-params.conv_window);
        prev_mean = log_mean(eval_res.ll_train, useinds);
        conv_meas =  abs((cur_mean-prev_mean)/(prev_mean));
        % TODO make option for convergence: vlb vs train likelihood
        % conv_meas = abs(aux.vlb(aux.vlb_itend(end-1)) - aux.vlb(aux.vlb_itend(end)))/(-1*aux.vlb(aux.vlb_itend(end)));
        if params.chk_conv && conv_meas <= params.conv_thresh && blk_idx > size(Xfull, 1)
            converge = true;
        end
    else
        conv_meas = -1;
    end
    if params.it_vis
        generate_plots(model, aux, params);
    end
    
    fprintf('K: %-3i vlb: %6.3f conv: %0.5f itr time %0.3f, tll %0.3e \n', model.K, aux.vlb(end), conv_meas, times(ii),eval_res.ll(ii));
    ii = ii + 1;
    if params.use_runtime
        if params.max_runtime < tot_runtime
            break;
        end
    end
    stime=tic;
end
eval_res.l2 = eval_res.l2(~isnan(eval_res.l2));
eval_res.ll = eval_res.ll(~isnan(eval_res.ll));
aux.eval_res = eval_res;
aux.times = times(~isnan(times)); 
aux.comp_res = comp_res;
aux.final_its = ii -1;

