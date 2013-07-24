function gendata = gen_ibp_data(N, varargin)
% function gendata = gen_ibp_data(N, varargin)
%
% generate latent factor data for the linear-Gaussian model ZA, where Z is a
% binary matrix and A is the factors. The user has the option of supplying
% a cell array 'facts' that indicate the 1 entreis in a binary factor
% matrix, supplying the factors matrix A, or specifying that the factor
% matrix should be drawn from a Gaussian or Truncated Gaussian prior.
%
% By default, gen_ibp_data will generate the binary features data from
% Infnite Latent Feature Models and the Indian Buffet Process 
% Griffiths & Ghahramani (2005)
%
% OUTPUT
% gendata: structure describing all components used for data generation
%
% REQUIRED
% N: number of data points to generate
%
% OPTIONS
% facts: cell-array indicating the 1 entries in a binary factor matrix A
% (default: {})
%
% sigX: standard deviation of data noise in linear gaussian model (default:
% 1.0)
%
% A: latent factor matrix (default [])
%
% D: dimensionality of generated data (not used if A or facts provided)
% (default: 36, corresponding to the default binary images data)
%
% Zprob: independent Bernoulli probability of each observation using a given feature, 
% set to -1 to generate Z from IBP (must specify alpha)
%
% K: Number of features, set to -1 to generate random number from prior, 
% else rejection sampling is used with IBP prior (which may take awhile)
%
% alpha: IBP parameter controlling expected number of features (ignored unless Zprob = -1)
%
% binary_fact_multiplicity: If A or facts is not provided, 
% each binary feature in the A matrix occupies potentially overlapping
%
% binary_fact_multiplicity: relative multiplicity of binary factors if randomly generated
%
% author: Colorado Reed (colorado.j.reed at gmail)
        
% extract parameters and prep for data generation
p = inputParser;
p.addRequired('N', @isnumeric);
p.addOptional('facts', {});
p.addOptional('D', 36, @isnumeric);
p.addOptional('sigX', 1.0, @isnumeric);
p.addOptional('alpha', 1.0, @isnumeric);
p.addOptional('A', []);
p.addOptional('K', 4);
p.addOptional('binary_fact_multiplicity', 1/3, @isnumeric);
p.addOptional('Zprob', 0.5, @isnumeric); 
p.parse(N, varargin{:});

gendata.N = N;
gendata.facts = p.Results.facts;
gendata.sigX = p.Results.sigX;
gendata.D = p.Results.D;
gendata.Zprob = p.Results.Zprob;
gendata.K = p.Results.K;
gendata.alpha = p.Results.alpha;
binary_fact_multiplicity = p.Results.binary_fact_multiplicity;

% generate Z matrix
if gendata.Zprob == -1
    [gendata.Z, gendata.K] = ibp_generate(gendata.N, gendata.alpha,gendata.K);
else 
    gendata.Z = rand(gendata.N, gendata.K) < gendata.Zprob; 
end

if isempty(p.Results.A)
    % assign default features if no features were provided
    if isempty(gendata.facts) && gendata.K == 4
        gendata.facts = {[2, 7, 8, 9, 14], [4, 5, 6, 10, 12, 16, 17, 18], [19, 25, 26, 31, 32, 33], [22, 23, 24, 29, 35]};
    else
        nfacts = gendata.K;
        gendata.facts = {};
        for ii=1:nfacts
            tmp = randperm(gendata.D);
            gendata.facts{ii} = tmp(1:ceil(gendata.D*binary_fact_multiplicity)); % TODO make this an option
        end
    end
    gendata.K = length(gendata.facts);
    gendata.A = zeros(gendata.K, gendata.D);
    for i=1:gendata.K
        gendata.A(i, gendata.facts{i}) = 1;
    end
else
    gendata.A = p.Results.A;
    gendata.K = size(gendata.A,2);
end

gendata.X = gendata.Z*gendata.A + gendata.sigX.*randn(gendata.N, gendata.D);

