function [logP_in logP_out] = uncoll_llhood( z , x , sigma_x , mA , pA , param_set, unnorm_set )
% function [logP_in logP_out] = uncoll_llhood( z , x , sigma_x , mA , pA , param_set, unnorm_set )
% 
% Computes the likelihood of an uncollapsed linear-Gaussian IBP model
%
% INPUT
% z: latent feature matrix
% x: input data matrix
% sigma_x: data std of for linear-Gaussian IBP model
$ mA: mean values for the linear-Gaussian IBP model
% pA: cov values for the linear-Gaussian IBP model
% param_set: param_set for the linear-Gaussian IBP model
% unnorm_set: set of unnormalization parameters incase data has been scaled/tranlated
%
% OUTPUT
% logP_in: 1-param_Set.test_mask of likelihood values
% logP_out: likelihood of complement of logP_in values
%
% Code derived by Colorado Reed from Finale Doshi-Velez's IBP code (http://people.csail.mit.edu/finale/new-wiki/doku.php?id=publications_posters_presentations_code)

% center the data
c_data = x - z * mA;

% unnormalize the data
if exist('unnorm_set','var') % we don't need to translate the data!
    N = size(c_data,1);
    c_data = c_data.*repmat(unnorm_set{2},N,1); %+ repmat(unnorm_set{1},N,1);
    if length(unnorm_set) > 2
        c_data = (c_data)./repmat(unnorm_set{4},N,1);
    end
end

% get p_data
p_data = sigma_x^2 * ones( size( x ) );
if ~isempty( pA )
    for d = 1:param_set.dim_setting_count
        p_data( : , param_set.d_ind_set{ d } ) = ...
            p_data( : , param_set.d_ind_set{ d } ) ... 
            + repmat( diag( z * pA{ d } * z' ) , [ 1 param_set.d_count_set( d ) ] ); 
    end
end

% compute each ll
ll = -1/2 * log( 2 * pi ) - 1/2 * log( p_data ) - 1/2 * c_data.^2 ./ p_data;     
% sum up the appropriate parts
logP_in = sum( sum( ll .* ( 1 - param_set.test_mask ) ) );
logP_out = sum( sum( ll .* param_set.test_mask ) );      