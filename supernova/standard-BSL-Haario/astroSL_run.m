
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


numsim= 100;  % was 1500
numattempts = 1;
shrinkage = 1; % shrinkage = 1 if we wish to regularize the SL covariance, else 0
if shrinkage
    shrink_param = 0.95; % must be in (0,1] % was 0.95
end

% bigtheta_start = parbase;
bigtheta_start = bigtheta_true;

Haario_mcmciter = 11200;
% proposal_cov = 1.0e-03 * [0.0717   -0.0102;
%                           -0.0102    0.3806];
% expandcov = 2600;
Haario_proposal = diag([0.01, 0.01].^2);
%Haario_proposal = expandcov*proposal_cov;  % inherit the proposal covariance matrix
Haario_length_CoVupdate = 50;
Haario_burnin = 200;
if mod(Haario_burnin,2)>0
    error('BURNIN must be even.')
end
chains = bslmcmc(bigtheta_start,parmask,parbase,s_obs,nobs,nbin,numsim,Haario_mcmciter,Haario_proposal,Haario_burnin,Haario_length_CoVupdate,shrinkage,shrink_param);
filename = sprintf('chains');
save(filename,'chains','-ascii')


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
