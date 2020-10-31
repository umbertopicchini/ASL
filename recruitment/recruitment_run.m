%rng(872223430)
rng(100)

r = 0.4;
k = 50;
alpha = 0.09;
beta = 0.05;

nobs = 300;  % only the last 250 will be actually used.

bigtheta_true =[r,k,alpha,beta];
% simulate "observed" summaries
s_obs = recruitment_simsummaries(bigtheta_true,1);


parbase = [  0.8     65    0.05   0.07];  
%parbase = [  1       75    0.02   0.07]; 
parmask = [   1       1       1    1   ];

bigtheta_start = parbase;

numsim= 200;  % model simulations per iteration
mcmciter= 10000; % number of iterations using ASL
sd_randomwalk = [0.005, 0.5, 0.001, 0.001];  % std deviation for Gaussian random walk during burnin
burnin = 300;  
robust = 0; % if 1 uses a procedure robust against lack-of Gaussianity for the summaries. If 0 uses the standard BSL
if mod(burnin,2)>0
    error('BURNIN must be even.')
end
mcwm = 1; % if 1 uses Markov-chain-within-Metropolis during burnin
% settings for adaptive SL 
forgetting = 0;
verbose = 1;
expandcov = 5;  % initial value for the expansion factor beta
lengthcovupdate = 50; % frequency of update for the expansion factor in ASL

chains = aslmcmc(bigtheta_start,parmask,parbase,s_obs,nobs,numsim,mcmciter,sd_randomwalk,burnin,forgetting,verbose,expandcov,robust,mcwm,lengthcovupdate);
save('chains','chains','-ascii');