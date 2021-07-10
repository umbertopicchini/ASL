rng(100)

r = 0.4;
k = 50;
alpha = 0.09;
beta = 0.05;


bigtheta_true =[r,k,alpha,beta];
% simulate "observed" summaries
s_obs = recruitment_simsummaries(bigtheta_true,1);

%parbase = bigtheta_true;
%parbase = [  0.8      65    0.05   0.07];  % startset 1
parbase = [  1       75    0.02   0.07]; % startset 2
parmask = [   1       1       1    1   ];

bigtheta_start = parbase;

numsim= 200;  
mcmciter= 2000;
sd_randomwalk = [0.005, 0.5, 0.001, 0.001];  % was [0.01, 0.1, 0.001, 0.001]
burnin = 300;  % must be even
length_CoVupdate = 50;  % frequency of covariance update for Haario's method 
robust = 1; % if 1 uses a procedure robust again lack-of Gaussianity for the summaries. If 0 uses the standard BSL
mcwm = 1;
if mod(burnin,2)>0
    error('BURNIN must be even.')
end

chains = bslmcmc_robust(bigtheta_start,parmask,parbase,s_obs,numsim,mcmciter,sd_randomwalk,burnin,length_CoVupdate,robust,mcwm);
save('chains','chains','-ascii')
