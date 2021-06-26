
rng(1234)  % was rng(1234)

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
numattempts = 5;
numgroups = 100; % numgroups must be such that mod(numsim,numgroups)==0
Haario_mcmciter = 3500;
Haario_proposal = diag([0.025, 0.025, 0.025, 0.025].^2);
Haario_length_CoVupdate = 30;
Haario_burnin = 200;
mcwm = 0; % must be 0 or 1. If mcwm==1 it will use MCWM during the burnin. 

if mod(Haario_burnin,2)>0
    error('BURNIN must be even.')
end

y = gk_rnd(theta_true,randn(nobs,1));

%thetastart = [1.6, 1.6, 1.6, 0];
%thetastart = [logA, logB, logg, logk];
% thetastart = zeros(numattempts,4);
% thetastart(1,:) = [2, 2, 1, 0.2];
% thetastart(2,:) = [1.6, 1.6, 1, 0];
% thetastart(3,:) = [1.6, 0.5, 0.5, 0];
 thetastart = [2, 2, 1, 0.2];

for attempt= 1:numattempts
    attempt

chains_withHaario = cslmcmc(thetastart,y,numsim,Haario_mcmciter,Haario_proposal,Haario_burnin,Haario_length_CoVupdate,numgroups,mcwm);
% now start a regular adaptive MCMC (Haario et al)
MCMC = chains_withHaario;
filename = sprintf('chains_attempt%d_numgroups_%d',attempt,numgroups);
save(filename,'MCMC','-ascii')
end

