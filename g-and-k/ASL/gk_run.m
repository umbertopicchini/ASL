
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


numsim= 1000;  % was 1500
mcmciter= 3500;
sd_randomwalk = [0.025, 0.025, 0.025, 0.025];
burnin = 200;  % must be even
Haario_length_CoVupdate = 30;
randomwalk_iterstart = 500; % the first post-burnin iteration when random-walk metropolis kicks-in
verbose = 1;
mcwm=1;  % if mcwm==1 we use BSL with MCWM sampling during the burnin iterations (NOT afterwards). If mcwm==0 we use standard BSL MCMC during burnin
numattempts = 5;

if mod(burnin,2)>0
    error('BURNIN must be even.')
end
if randomwalk_iterstart <= burnin
    error('RANDOMWALK_ITERSTART should be larger than burnin')
end
if burnin < Haario_length_CoVupdate
    error('BURNIN must be larger than Haario_length_CoVupdate')
end

y = gk_rnd(theta_true,nobs,1);


%thetastart = [logA, logB, logg, logk];
thetastart = [2, 2, 1, 0.2];
% thetastart = zeros(numattempts,4);
% thetastart(1,:) = [2, 2, 1, 0.2];
% thetastart(2,:) = [1.6, 1.6, 1, 0];
% thetastart(3,:) = [1.6, 0.5, 0.5, 0];


for attempt= 1:numattempts
    attempt
[chains,proposal_cov] = aslmcmc(thetastart,y,numsim,mcmciter,sd_randomwalk,Haario_length_CoVupdate,burnin,randomwalk_iterstart,verbose,mcwm);

filename = sprintf('chains_attempt%d',attempt);
save(filename,'chains','-ascii')
end

