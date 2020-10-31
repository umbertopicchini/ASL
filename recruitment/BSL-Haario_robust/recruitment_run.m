rng(100)

r = 0.4;
k = 50;
alpha = 0.09;
beta = 0.05;

nobs = 300;  % only the last 250 will be actually used.

bigtheta_true =[r,k,alpha,beta];
% simulate "observed" summaries
s_obs = recruitment_simsummaries(bigtheta_true,1);

%parbase = bigtheta_true;
%parbase = [  0.8      65    0.05   0.07];  
parbase = [  1       75    0.02   0.07];
parmask = [   1       1       1    1   ];

bigtheta_start = parbase;

numsim= 200;  % model simulations per iteration
mcmciter= 5000; % total number of MCMC iterations
sd_randomwalk = [0.005, 0.5, 0.001, 0.001];  % standard deviations for random walk during burnin 
burnin = 300;  % must be even
length_CoVupdate = 50;  % frequency of covariance update for Haario's method 
robust = 1; % if 1 uses a procedure robust again lack-of Gaussianity for the summaries. If 0 uses the standard BSL
mcwm = 1;  % if 1 uses Markov-chain-within-Metropolis during burnin
if mod(burnin,2)>0
    error('BURNIN must be even.')
end

chains = bslmcmc_robust(bigtheta_start,parmask,parbase,s_obs,nobs,numsim,mcmciter,sd_randomwalk,burnin,length_CoVupdate,robust,mcwm);
save('chains','chains','-ascii')