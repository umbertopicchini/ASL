
rng(1234)  % was rng(1234)

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

% %         logom      ok      w0   wa   h0
% parbase = [ -0.11     0    -0.5    0    0.7];
% parmask = [   1       0       1    0     0 ];

%         logom      ok      w0   wa   logh0
parbase = [ -0.11     0    -0.5    0   logh0];
parmask = [   1       0       1    0     0  ];

%bigtheta_start = parbase;


numsim= 100;  
mcmciter= 11200; 
sd_randomwalk = [0.01, 0.01];  % was 0.03 0.03 0.03 
burnin = 200;  % must be even
randomwalk_iterstart = 500; % the first post-burnin iteration when random-walk metropolis kicks-in
length_CoVupdate = 50;
numattempts = 1;
numgroups = 10; % numgroups must be such that mod(numsim,numgroups)==0
shrinkage = 1; % shrinkage = 1 if we wish to regularize the SL covariance, else 0
if shrinkage
    shrink_param = 0.95; % must be in (0,1] % was 0.95
end
if mod(burnin,2)>0
    error('BURNIN must be even.')
end

verbose = 1;
thetastart = zeros(numattempts,2);
thetastart(1,:) = [ -0.11  -0.5 ];  % om=0.90, w0= -0.5
%thetastart(2,:) = [ -0.11  0];   % om = 0.90; w0= 0
%thetastart(3,:) = [-2.30  0];    % om = 0.1; w0 = 0


for attempt= 1:numattempts
    attempt
    bigtheta_start = param_unmask(thetastart(attempt,:),parmask,parbase);
    [chains,proposal_cov] = aslmcmc(bigtheta_start,parmask,parbase,s_obs,nobs,nbin,numsim,mcmciter,sd_randomwalk,length_CoVupdate,burnin,randomwalk_iterstart,numgroups,shrinkage,shrink_param,verbose);
  %  chains = bslmcmc_tuned9_withappended_Haario(bigtheta_start,parmask,parbase,s_obs,nobs,nbin,numsim,mcmciter,sd_randomwalk,burnin,numgroups,shrinkage,shrink_param,forgetting,verbose,expandcov,haario_iter,length_CoVupdate);
    filename = sprintf('chains_attempt%d_numgroups=%d',attempt,numgroups);
    save(filename,'chains','-ascii')
end

figure
subplot(2,2,1)
plot(exp(chains(:,1))) % exponentiate logom --> om
hline(om)
subplot(2,2,2)
plot(chains(:,2))
hline(w0)
% subplot(2,2,3)
% plot(exp(chains(:,3))) % exponentiate logh0 --> h0
% hline(h0)
