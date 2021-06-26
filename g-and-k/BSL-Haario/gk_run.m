
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
% mcmciter= 5200;
% sd_randomwalk = [0.025, 0.025, 0.025, 0.025];
% burnin = 200;  % must be even
% length_CoVupdate = 1;  % was 100
% forgetting = 0;
% verbose = 1;
% mcwm=1;
% expandcov = 4;
numattempts = 5;
mcwm=1;


% if mod(burnin,2)>0
%     error('BURNIN must be even.')
% end

y = gk_rnd(theta_true,nobs,1);

%thetastart = [logA, logB, logg, logk];
%  thetastart = zeros(numattempts,4);
%  thetastart(1,:) = [2, 2, 1, 0.2];
% thetastart(2,:) = [1.6, 1.6, 1, 0];
% thetastart(3,:) = [1.6, 0.5, 0.5, 0];

thetastart = [2, 2, 1, 0.2];

for attempt= 1:numattempts
    attempt
%[chains,proposal_cov] = bslmcmc_tuned10(thetastart(attempt,:),y,numsim,mcmciter,sd_randomwalk,burnin,length_CoVupdate,forgetting,verbose,mcwm,expandcov);
Haario_mcmciter = 3200;
% proposal_cov = 1.0e-03 *[0.000192907520794   0.000617342066926  -0.000600313040005  -0.000890376859585;
%    0.000617342066926   0.009415006095552   0.003804927540416  -0.008105040399638;
%   -0.000600313040005   0.003804927540416   0.016882711948843  -0.007988429991122;
%   -0.000890376859585  -0.008105040399639  -0.007988429991122   0.241854856626190];
%expandcov = 400;
%Haario_proposal = expandcov*proposal_cov;  % inherit the proposal covariance matrix
Haario_proposal = diag([0.025, 0.025, 0.025, 0.025].^2);
Haario_length_CoVupdate = 30;
Haario_burnin = 200;
chains_withHaario = bslmcmc(thetastart,y,numsim,Haario_mcmciter,Haario_proposal,Haario_burnin,Haario_length_CoVupdate,mcwm);
% now start a regular adaptive MCMC (Haario et al)
MCMC = chains_withHaario;
filename = sprintf('chains_attempt%d',attempt);
save(filename,'MCMC','-ascii')
end

