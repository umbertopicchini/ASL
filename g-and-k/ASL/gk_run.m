
rng(1234) 

% ground-truth parameters
A = 3.0;
B = 1.0;
g = 2.0;
k = 0.5;

logA = log(A);
logB = log(B);
logg = log(g);
logk = log(k);


nobs = 1000;  % data sample size
theta_true = [logA, logB, logg, logk];  % ground-truth parameters


numsim= 1000;  % model simulations per iteration
mcmciter= 5200; % total number of MCMC iterations
sd_randomwalk = [0.025, 0.025, 0.025, 0.025]; % standard deviations for random walk during burnin (on parameters log-scale)
burnin = 200;  % must be even
verbose = 1;
mcwm=1;  % if 1 uses Markov-chain-within-Metropolis during burnin
expandcov = 10;  % starting value for the expansion factor beta
numattempts = 3;

if mod(burnin,2)>0
    error('BURNIN must be even.')
end

y = gk_rnd(theta_true,nobs,1);


%thetastart = [logA, logB, logg, logk];
thetastart = zeros(numattempts,4);
thetastart(1,:) = [2, 2, 1, 0.2];
thetastart(2,:) = [1.6, 1.6, 1, 0];
thetastart(3,:) = [1.6, 0.5, 0.5, 0];


for attempt= 1:numattempts
    attempt
[chains,proposal_cov] = aslmcmc(thetastart(attempt,:),y,numsim,mcmciter,sd_randomwalk,burnin,verbose,mcwm,expandcov);

filename = sprintf('chains_attempt%d',attempt);
save(filename,'chains','-ascii')
end

