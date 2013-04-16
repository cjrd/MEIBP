% Test script for MEIBP inference
%
% C. Reed & Z. Ghahramani (2013)
% Scaling the Indian Buffet Process via Submodular Maximization
% http://www.arxiv.org/abs/1304.3285
%
% author: Colorado Reed, gmail address: colorado.j.reed

addpath('utils');

% visualize the inference demo?
do_visualization = true;

% set random seed (Change me to examine different data/initializations)
sk = [];

% data: example binary image data with 4 latent features
N = 2000;
sigX = 0.5; % high data noise (see visualization)
gmodel = gen_ibp_data(N, 'sigX', sigX);

% 0-min the data

% mask some test data
test_mask = zeros(size(gmodel.X)); % can make sparse for large data (advantageous for N > 10^6)
test_mask(N-round(N/4), 1:round(gmodel.D)/4); 

%-----------------------%
%      Parameters       %
%-----------------------%

% initialization parameters
kinit = 20; % set initial upper-bound on number of latent features (true number is 4)
Z_init_per = 1/3; % percentage of Z entries to randomly set equal to 1 (random initialization)

% specify which IBP prior to use
use_lof = false; % false uses shifted equivalence classes

% model hyperparameters % use reasonable heuristics
alpha = (1 + kinit)/(1+sum(1./(1:N)));
sigX = 1;
sigA = 1;

% algorithm options
num_restarts = 1; % number of random restarts
rand_n_order = false; % update instances sequentially or randomly?
rand_k_order = false; % update features sequentially or randomly?

% convergence options
use_max_runtime = true;
max_runtime = 3600; % max runtime in seconds
chk_conv = true; % check for convergence
conv_method = 'vlb'; % choose 'vlb' or 'tll' [variational lower bound or training likelihood]
conv_thresh = 10e-4; % multiplicative difference in conv_method
conv_window = 2; % use average convergence criterion in an n-block window

% latent factor updates options % TODO implement option
update_a_each_zn = true; % update A after each Zn is optimized; slower but better much better results (especially with random initialization)
exa_uplag = 2; % how often to update the A expectations (every n A_i feature updates); generally higher values decrease runtime and results

% other options
detailed_vlb = false; % keep track of all vlb changes (much slower but good for debugging)

% set a parameter struct to pass params/options into MEIBP
params = struct(...
    'Kinit', kinit,...
    'test_mask', test_mask,...
    'num_restarts', num_restarts,...
    'Zinit_per', Z_init_per,...
    'alpha', alpha,...
    'sigX', sigX,...
    'sigA', sigA,...
    'use_runtime', use_max_runtime,...
    'max_runtime', max_runtime,...
    'chk_conv', chk_conv,...
    'conv_method', conv_method,...
    'conv_thresh', conv_thresh,...
    'conv_window', conv_window,...
    'exa_uplag', exa_uplag,...
    'detailed_vlb', detailed_vlb,...
    'exch_ec', use_lof,...
    'rand_n_order', rand_n_order,...
    'rand_k_order', rand_k_order,...
    'update_a_each_zn',update_a_each_zn...
    );


%-----------------------%
%    MEIBP Inference    %
%-----------------------%
rr_res = meibp(gmodel.X, params);
% use the best result over the random restarts
[vlb, iloc] = max([rr_res.vlb]);
meibp_aux = rr_res(iloc).aux;
meibp_res = rr_res(iloc).model;


%%
if do_visualization
    %-----------------------%
    %    Visualization      %
    %-----------------------%
    close all;
    %% visualize the factors
    fnum = 1;
    sqdim = sqrt(size(gmodel.A,2));
    colormap(gray)
    figure(fnum)
    fnum = fnum + 1;
    for i=1:4
        subplot(1,4,i)
        imagesc(reshape(gmodel.A(i,:), sqdim, sqdim))
        if i ==1
            title('true features')
        end
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        grid off
    end
    
    %% visualize some data
    dvals = randperm(N);
    dvals = dvals(1:36);
    figure(fnum)
    fnum = fnum + 1;
    colormap(gray)
    for i=1:36
        subplot(6,6,i)
        imagesc(reshape(gmodel.X(dvals(i),:), sqdim, sqdim))
        if i==1
            title('input data');
        end
        grid off
    end
    
    %% visualize the learned factors
    figure(fnum)
    colormap(gray)
    fnum = fnum + 1;
    mks = sum(meibp_res.Z);
    [tmp,pop_feats] = sort(mks,'descend');
    ct = 1;
    for i=pop_feats(1:min(9,size(meibp_aux.Ex_A,1)))
        subplot(3,3,ct);
        imagesc(reshape(meibp_aux.Ex_A(i,:), sqdim, sqdim), [0,1]);
        colorbar;
        title(['m_k =  ' num2str(mks(i))]);
        ct = ct +1;
        grid off
    end
    
    %% visualize Z
    figure(fnum)
    fnum = fnum + 1;
    colormap(gray)
    fnum = fnum + 1;
    imagesc(meibp_res.Z);
    title('Z')
    grid off;
    
    %% plot the variational lower bound
    figure(fnum)
    fnum = fnum + 1;
    plot(meibp_aux.vlb, 'o-k','linewidth',2);
    xlabel('iteration');
    ylabel('VLB')
    
end
