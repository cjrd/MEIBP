% Run from MEIBP directory

addpath('utils');

% set random seed (Change me to examine different data/initializations)
rng(1)

% data: example binary image data with 4 latent features
N = 1000;
sigX = 0.5;
gmodel = gen_ibp_data(N, 'sigX', sigX); 

% 0-min the data

% mask some test data
test_mask = zeros(size(gmodel.X)); % can make sparse for large data (advantage for N < 10^5)

%-----------------------%
%      Parameters       %
%-----------------------%

% initialization parameters
kinit = 10; % set initial upper-bound on number of latent features (true number is 4)
Z_init_per = 0.3; % percentage of Z entries to randomly set equal to 1 (random initialization)
% TODO BP++ init or init from prior

% specify which IBP prior to use
use_lof = false; % false uses shifted ec (faster)

% model hyperparameters
alpha = 1;
sigX = 1;
sigA = 1;

% algorithm options
add_data_in_blocks = false; % TODO BROKEN
data_block_size = 250;
tempering = false;
add_features = false; % greedily add features % TODO BROKEN

% convergence options
use_max_runtime = true;
max_runtime = 3600; % max runtime in seconds
chk_conv = true; % check for convergence
conv_method = 'vlb'; % choose 'vlb' or 'tll' [variational lower bound or training likelihood]
conv_thresh = 10e-4; % multiplicative difference in conv_method
conv_window = 2; % use average convergence criterion in a n-block window

% latent factor updates options
update_a_each_zn = true; % update A after each Zn is optimized; slower but better results
% TODO!
exa_uplag = 2; % how often to update the A expectations (every n A_i feature updates); generally higher values decrease runtime and results

% other options
detailed_vlb = false; % keep track of all vlb changes (much slower but good for debugging)

% set a parameter struct to pass params/options into MEIBP
params = struct(...
    'Kinit', kinit,...
    'test_mask', test_mask,...
    'Zinit_per', Z_init_per,...
    'alpha', alpha,...
    'sigX', sigX,...
    'sigA', sigA,...
    'data_blocks', add_data_in_blocks,...
    'data_block_size', data_block_size,...
    'use_runtime', use_max_runtime,...
    'max_runtime', max_runtime,...
    'chk_conv', chk_conv,...
    'conv_thresh', conv_thresh,...
    'conv_window', conv_window,...
    'exa_uplag', exa_uplag,...
    'tempering', tempering,...
    'add_features', add_features,...
    'detailed_vlb', detailed_vlb,...
    'exch_ec', use_lof...
    );


%-----------------------%
%    MEIBP Inference    %
%-----------------------%
[meibp_res, meibp_aux] = meibp(gmodel.X, params);


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
    subplot(2,2,i)
    imagesc(reshape(gmodel.A(i,:), sqdim, sqdim))
    if i ==1
        title('true features')
    end
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