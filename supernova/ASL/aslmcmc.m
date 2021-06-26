function [MCMC,proposal_cov]= aslmcmc(bigthetastart,parmask,parbase,sobs,nobs,nbin,numsim,R_mcmc,step_rw,length_CoVupdate,burnin,randomwalk_iterstart,numgroups, shrinkage, gamma,verbose)
% thetastart: starting parameter values
% numsim: number of simulated datasets at teach mcmc iteration
% R_mcmc: number of mcmc iterations
% standard deviations for the diagonal covariance matrix of MRW


if mod(numsim,2)>0
    error('NUMSIM should be an even integer. Also, it should be a multiple of NUMGROUPS')
end
if mod(numsim,numgroups)>0
    error('NUMGROUPS should be an even integer such that NUMSIM is a multiple of NUMGROUPS.')
end
if numgroups >= numsim
    error('NUMGROUPS should be smaller than NUMSIM.')
end

thetastart = param_mask(bigthetastart,parmask);

% prepare columns of indeces, useful for the correlated blocking strategy
indeces = [1:numsim]; % all integer indeces 1,2,...,numsim
blocks = reshape(indeces, numsim/numgroups, numgroups); % block is now a [numsim/numgroups] x numgroups matrix where all the indeces created above are arranged into numgroups columns

MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

bigtheta_old = param_unmask(theta_old,parmask,parbase);

dtheta = length(thetastart);
dsobs = length(sobs);
%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::

unif_old = rand(nobs,numsim);
if shrinkage
    [loglik_old,~,~,simsum] = astroSL_synlik_shrinkage(bigtheta_old,sobs,nobs,nbin,numsim,unif_old, shrinkage, gamma);
else
    [loglik_old,~,~,simsum] = astroSL_synlik(bigtheta_old,sobs,nobs,nbin,numsim,unif_old);
end

accepted_simsum = simsum';
accepted_thetasimsum = zeros(R_mcmc,dtheta+dsobs);
accepted_thetasimsum(1,:) = [theta_old,mean(accepted_simsum,1)];

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at old parameters
prior_old =  astroSL_prior(theta_old);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

% initial (diagonal) covariance matrix for the Gaussian proposal
cov_current = step_rw.^2 .* eye(length(thetastart));
bigtheta_old = param_unmask(theta_old,parmask,parbase);
if isinf(loglik_old) || (prior_old==0) || isnan(loglik_old)
  loglik_old = -1e300;
  warning("The initial proposal is not admissible. Assigning the loglikelihood a value -1e300...")
%   theta_old_attempt = mvnrnd(theta_old,cov_current);
%   bigtheta_old_attempt = param_unmask(theta_old_attempt,parmask,parbase);
%   unif_old = rand(nobs,numsim);
%   if shrinkage
%       [loglik_old,~,~,simsum] = astroSL_synlik_shrinkage(bigtheta_old,sobs,nobs,nbin,numsim,unif_old, shrinkage, gamma);
%   else
%       [loglik_old,~,~,simsum] = astroSL_synlik(bigtheta_old,sobs,nobs,nbin,numsim,unif_old);
%   end
%   prior_old =  astroSL_prior(theta_old_attempt);
end

% let's store the following once more, just in case we entered the WHILE loop above
% theta_old = theta_old_attempt;
% bigtheta_old = param_unmask(theta_old,parmask,parbase);
% accepted_simsum = simsum';
% accepted_thetasimsum = zeros(R_mcmc,dtheta+dsobs);
% accepted_thetasimsum(1,:) = [theta_old,accepted_simsum(1,:)];


% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);
bigtheta = param_unmask(theta,parmask,parbase);
select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
unif = unif_old(:);  % make the matrix of random draws a vector to simplify things
unif(select_indeces) = rand(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same 
unif = reshape(unif,nobs,numsim); % go back to original dimensions
if shrinkage
    [loglik,~,~,simsum] = astroSL_synlik_shrinkage(bigtheta,sobs,nobs,nbin,numsim,unif, shrinkage, gamma);
else
    [loglik,~,~,simsum] = astroSL_synlik(bigtheta,sobs,nobs,nbin,numsim,unif);
end

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at proposed parameters
prior =  astroSL_prior(theta);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if log(rand) < loglik-loglik_old +log(prior)-log(prior_old)
  % here we accept our proposal theta
  MCMC(2,:) = theta;
  loglik_old = loglik;
  theta_old = theta;
  prior_old = prior;
  unif_old = unif;
  accepted_simsum = simsum';
  mean_accepted_simsum = mean(accepted_simsum,1);
  accepted_thetasimsum(2,:) = [theta_old,mean_accepted_simsum];
else
  % reject proposal
  MCMC(2,:) = theta_old;
  mean_accepted_simsum = mean(accepted_simsum,1);
  accepted_thetasimsum(2,:) = [theta_old,mean_accepted_simsum];
end

% we execute an MCMC for only burnin/2 iterations only in otder to compute
% covariances between theta and summary statistics. It is unsafe to compute
% also other quantities so early in the MCMC chain and instead wait  a little more, 
% until the chain reaches more promising areas 

for mcmc_iter = 3:burnin/2 

   theta = mvnrnd(theta_old,cov_current);
   bigtheta = param_unmask(theta,parmask,parbase);
   
   % update pseudo-random numbers in a randomly selected block of indeces
   select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
   select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
   unif = unif_old(:);  % make the matrix of random draws a vector to simplify things
   unif(select_indeces) = rand(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same as in the previous iteration
   unif = reshape(unif,nobs,numsim); % go back to original dimensions
   
   if shrinkage
      [loglik,~,~,simsum] = astroSL_synlik_shrinkage(bigtheta,sobs,nobs,nbin,numsim,unif, shrinkage, gamma);
   else
      [loglik,~,~,simsum] = astroSL_synlik(bigtheta,sobs,nobs,nbin,numsim,unif);
   end

   % evaluate priors at proposed parameters
   prior =  astroSL_prior(theta);
   
  if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
     MCMC(mcmc_iter,:) = theta;
     loglik_old = loglik;
     theta_old = theta;
     prior_old = prior;
     unif_old = unif;
     accepted_simsum = simsum';
     mean_accepted_simsum = mean(accepted_simsum,1);
     accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean_accepted_simsum];
     if verbose
        fprintf('\nMCMC iter %d: acceptance...',mcmc_iter)
     end
  else
     MCMC(mcmc_iter,:) = theta_old;
     sampled_indeces = randsample(numsim,numsim,true);
     accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean(accepted_simsum(sampled_indeces,:),1)];
  end
end

lastCovupdate = 0;
num_proposal = 0;
accept_proposal = 0;


for mcmc_iter = burnin/2+1:R_mcmc

  if mcmc_iter == burnin
%        if burnin >= length_CoVupdate
%           lastCovupdate = burnin-length_CoVupdate;
%        end
       lastCovupdate = randomwalk_iterstart-length_CoVupdate;
       cov_last = cov_current;
       % create the first instance of the guidd mean for the proposal
        mean_theta = mean(accepted_thetasimsum(burnin/2:mcmc_iter,1:dtheta));
        mean_simsum = mean(accepted_thetasimsum(burnin/2:mcmc_iter,dtheta+1:end));
        all_cov_thetasimsum = cov(accepted_thetasimsum(burnin/2:mcmc_iter,:));
%        mean_theta = mean(accepted_thetasimsum(1:mcmc_iter,1:dtheta));
%        mean_simsum = mean(accepted_thetasimsum(1:mcmc_iter,dtheta+1:end));
%        all_cov_thetasimsum = cov(accepted_thetasimsum(1:mcmc_iter,:));
       cov_theta = all_cov_thetasimsum(1:dtheta,1:dtheta);
       cov_simsum = all_cov_thetasimsum(dtheta+1:end,dtheta+1:end);
       cov_thetasimsum = all_cov_thetasimsum(1:dtheta,dtheta+1:end);
       cov_simsumtheta = cov_thetasimsum';
       proposal_mean = mean_theta' + cov_thetasimsum * (cov_simsum \(sobs-mean_simsum'));
       proposal_mean =  proposal_mean';  % must be a row vector when passed to mvnrnd()
       proposal_cov = cov_theta - cov_thetasimsum * (cov_simsum \ cov_simsumtheta);
       proposal_cov = (proposal_cov + proposal_cov')/2; % trick to make it symmetric
       [~, notposdef] = cholcov(proposal_cov);
       if isnan(notposdef) || notposdef>0
          [L, DMC, P] = modchol_ldlt(proposal_cov); 
           proposal_cov = P'*L*DMC*L'*P;
       end
   end 
   if mcmc_iter < burnin
       theta = mvnrnd(theta_old,cov_current);  % Gaussian random walk
   else
               if mcmc_iter >= randomwalk_iterstart
                   % do random walk metropolis
                   if (mcmc_iter == lastCovupdate+length_CoVupdate) % adapt covariance a-la Haario et al 2001
                      fprintf('\nMCMC iteration -- adapting Haario covariance...')
                      covupdate = cov(MCMC(burnin:mcmc_iter-1,1:end));
                      % compute equation (1) in Haario et al.
                      cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-8 * eye(length(theta)) ;
                      lastCovupdate = mcmc_iter;
                      fprintf('\nMCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
                      accept_proposal=0;
                      num_proposal=0;
                      MCMC_temp = MCMC(1:mcmc_iter-1,:);
                      save('THETAmatrix_temp','MCMC_temp');
                   end
                   theta = mvnrnd(theta_old,cov_current);
               else
                   % use guided proposals
                   theta = mvnrnd(proposal_mean,proposal_cov);
                   cov_current = proposal_cov;
               end
               cov_last = cov_current;       
   end 
   
   bigtheta = param_unmask(theta,parmask,parbase);
   
   % update pseudo-random numbers in a randomly selected block of indeces
   select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
   select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
   unif = unif_old(:);  % make the matrix of random draws a vector to simplify things
   unif(select_indeces) = rand(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same as in the previous iteration
   unif = reshape(unif,nobs,numsim); % go back to original dimensions
   if shrinkage
      [loglik,~,~,simsum] = astroSL_synlik_shrinkage(bigtheta,sobs,nobs,nbin,numsim,unif, shrinkage, gamma);
   else
      [loglik,~,~,simsum] = astroSL_synlik(bigtheta,sobs,nobs,nbin,numsim,unif);
   end

   % evaluate priors at proposed parameters
   prior =  astroSL_prior(theta);


   if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
      if verbose
         fprintf('\nMCMC iter %d: proposal accepted...',mcmc_iter)
      end
      MCMC(mcmc_iter,:) = theta;
      loglik_old = loglik;
      theta_old = theta;
      prior_old = prior;
      unif_old = unif;
      accepted_simsum = simsum';
      mean_accepted_simsum = mean(accepted_simsum,1);
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean_accepted_simsum];
      if verbose
         fprintf('\nMCMC iter %d: acceptance...',mcmc_iter)
      end
   else
      MCMC(mcmc_iter,:) = theta_old;
      sampled_indeces = randsample(numsim,numsim,true);
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean(accepted_simsum(sampled_indeces,:),1)];
    %  accepted_thetasimsum(mcmc_iter,:) = [theta_old,accepted_simsum(randi(numsim),:)];
   end 
   

  if (mcmc_iter >= burnin)
       mean_theta = mean(accepted_thetasimsum(burnin/2:mcmc_iter,1:dtheta));
       mean_simsum = mean(accepted_thetasimsum(burnin/2:mcmc_iter,dtheta+1:end));
       all_cov_thetasimsum = cov(accepted_thetasimsum(burnin/2:mcmc_iter,:));
%        mean_theta = mean(accepted_thetasimsum(1:mcmc_iter,1:dtheta));
%        mean_simsum = mean(accepted_thetasimsum(1:mcmc_iter,dtheta+1:end));
%        all_cov_thetasimsum = cov(accepted_thetasimsum(1:mcmc_iter,:));
       cov_theta = all_cov_thetasimsum(1:dtheta,1:dtheta);
       cov_simsum = all_cov_thetasimsum(dtheta+1:end,dtheta+1:end);
       cov_thetasimsum = all_cov_thetasimsum(1:dtheta,dtheta+1:end);
       cov_simsumtheta = cov_thetasimsum';
       proposal_mean = mean_theta' + cov_thetasimsum * (cov_simsum \(sobs-mean_simsum'));
       proposal_mean =  proposal_mean';  % must be a row vector when passed to mvnrnd()
       proposal_cov = cov_theta - cov_thetasimsum * (cov_simsum \ cov_simsumtheta);
       proposal_cov = (proposal_cov + proposal_cov')/2; % trick to make it symmetric
       [~, notposdef] = cholcov(proposal_cov);
       if isnan(notposdef) || notposdef>0
          [L, DMC, P] = modchol_ldlt(proposal_cov); 
           proposal_cov = P'*L*DMC*L'*P;
       end
   end
end

proposal_cov = cov_last; % this is only useful as it is returned as a output from this function

end