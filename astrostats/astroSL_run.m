
rng(1234)  

% ground-truth parameters
om = 0.3;
ok = 0;
w0 = -1;
wa = 0;
h0 = 0.7;

logom = log(om); % -0.52
logh0 = log(h0); % -0.357



nobs = 10000;  % data sample size
nbin = 20;
bigtheta_true = [logom, ok, w0, wa, logh0];  % ground-truth parameters

% simulate "observed" summaries
s_obs = astroSL_simsummaries(bigtheta_true,nobs,nbin,1,rand(nobs,1));

%:::: end of data generation :::::::::::::::::::::::::::::::::::::::::::

% set parameters starting values
%         logom      ok      w0   wa   logh0
parbase = [ -0.11     0    -0.5    0   logh0];
parmask = [   1       0       1    0     0  ];

numsim= 100; % the number of simulated summaries in SL 
mcmciter= 500;  % number of iterations using ASL
sd_randomwalk = [0.01, 0.01];  % std deviations for Gaussian random walks
burnin = 200;  % must be even
haario_iter = 1000;  % number of additional MCMC iterations using Haario's method, AFTER the already performed mcmciter iterations
length_CoVupdate = 20;  % frequency of covariance update for Haario's method 
numgroups = 10; % number of groups/blocks for correlated SL; numgroups must be such that mod(numsim,numgroups)==0
shrinkage = 1; % shrinkage = 1 if we wish to regularize the SL covariance, else 0
if shrinkage
    shrink_param = 0.95; % shrinkage parameter, must be in (0,1] 
end
if mod(burnin,2)>0
    error('BURNIN must be even.')
end
% settings for adaptive SL 
forgetting = 0;
verbose = 1;
expandcov = 4; % expansion factor for the ASL covariance matrix ("beta" in the paper) 

numattempts = 2;
thetastart = zeros(numattempts,2);
thetastart(1,:) = [ -0.11  -0.5 ];  % om=0.90, w0= -0.5
thetastart(2,:) = [ -0.11  0];   % om = 0.90; w0= 0

% this will run two separate runs, one after the other, using the 
% two sets of starting values above
for attempt= 1:2
    attempt
    bigtheta_start = param_unmask(thetastart(attempt,:),parmask,parbase);
    chains = bslmcmc_tuned9_withappended_Haario(bigtheta_start,parmask,parbase,s_obs,nobs,nbin,numsim,mcmciter,sd_randomwalk,burnin,numgroups,shrinkage,shrink_param,forgetting,verbose,expandcov,haario_iter,length_CoVupdate);
    filename = sprintf('chains_attempt%d',attempt);
    save(filename,'chains','-ascii')
end

figure
subplot(2,2,1)
plot(exp(chains(:,1))) % exponentiate logom --> om
hline(om)
subplot(2,2,2)
plot(chains(:,2))
hline(w0)

