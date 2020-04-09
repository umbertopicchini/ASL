rng(872223430)

r = 0.4;
k = 50;
alpha = 0.09;
beta = 0.07;

nobs = 300;  % only the last 250 will be actually used.

bigtheta_true =[r,k,alpha,beta];
% simulate "observed" summaries
s_obs = recruitment_simsummaries(bigtheta_true,1);


parbase = [  0.8      65    0.05   0.07];  
parmask = [   1       1       1    1   ];

bigtheta_start = parbase;

numsim= 2000;  % number of summaries simulated using SL
mcmciter= 300; % number of iterations using ASL
sd_randomwalk = [0.005, 0.5, 0.001, 0.001];  % std deviation for Gaussian random walk during burnin
burnin = 200;  % must be even
haario_iter = 2000;  % number of additional MCMC iterations using Hario's method, AFTER the mcmciter already done
length_CoVupdate = 20;  % frequency of covariance update for Haario's method 
numgroups = 1; % number of groups for correlated SL; numgroups must be such that mod(numsim,numgroups)==0
if numgroups>1
    error('For this case-study correlated random draws are not enabled.')
end
robust = 0; % if 1 uses a procedure robust again lack-of Gaussianity for the summaries. If 0 uses the standard BSL
if mod(burnin,2)>0
    error('BURNIN must be even.')
end
% settings for adaptive SL 
forgetting = 0;
verbose = 1;
expandcov = (2.38)^2/sum(parmask);  % ACTUALLY UNUSED HERE

% here ASL uses a multivariate Student's proposal
chains = bslmcmc_tuned10_Students(bigtheta_start,parmask,parbase,s_obs,nobs,numsim,mcmciter,sd_randomwalk,burnin,numgroups,forgetting,verbose,expandcov,haario_iter,length_CoVupdate,robust);
