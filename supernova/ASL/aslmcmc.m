function [MCMC,proposal_cov]= aslmcmc(bigthetastart,parmask,parbase,sobs,nobs,nbin,numsim,R_mcmc,step_rw,burnin,numgroups, shrinkage, gamma,forgetting,verbose,expandcov)
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
lastexpandcovupdate = 0;
acceptrate = 0;
numproposal = 0;
numaccept = 0;
stop_update_expandcov = 0;


for mcmc_iter = burnin/2+1:R_mcmc

  % theta = mvnrnd(theta_old,cov_current);
   if mcmc_iter <= burnin
       theta = mvnrnd(theta_old,cov_current);  % Gaussian random walk
   else
       numproposal = numproposal+1;
     %  proposal_cov
       % here we update the expandcov factor according to the current
       % acceptance rate
       if ~stop_update_expandcov
         if (mcmc_iter == burnin + 50) || (mcmc_iter == lastexpandcovupdate+50)
            if acceptrate > 0.20  % acceptance rate is currently > 20 percent
               expandcov = expandcov + 0.25*expandcov;  % increase expandcov by 25 percent
            elseif acceptrate < 0.15 % acceptance rate is currently < 15 percent
               expandcov = expandcov - 0.05*expandcov;
               expandcov = max(1,expandcov);
            end
           lastexpandcovupdate = mcmc_iter;
           acceptrate = numaccept/numproposal;
           if verbose
              fprintf('\nMCMC iter %d: acceptance rate %d...',mcmc_iter,acceptrate)
              fprintf('\nMCMC iter %d: expandcov %d...',mcmc_iter,expandcov)
           end
           % reset quantities
           numaccept = 0;
           numproposal = 0;
         end
       end
       % Gaussian independence sampler
       try
          theta = mvnrnd(proposal_mean,expandcov^2*proposal_cov + 1e-8*eye(length(theta))/length(theta_old));
       catch
          fix_covariance = expandcov^2*proposal_cov + 1e-8*eye(length(theta))/length(theta_old);
          trick_covariance = (fix_covariance'+fix_covariance)/2; % good old trick...
          theta = mvnrnd(proposal_mean,trick_covariance);
       end
%        if acceptrate < 0.25 && acceptrate > 0.2
%            stop_update_expandcov = 1;  % no more adaptation for expandcov from now on
%            fprintf('\n**expandcov=%d, WE STOPPED UPDATING expandcov**',expandcov)
%            save('expandcov.txt','expandcov','-ascii')
%            proposal_cov
%        end
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
      if mcmc_iter > burnin
         numaccept = numaccept+1;
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
   
  if (forgetting > 0) && (mcmc_iter > burnin) 
     forgetting = forgetting +1;  
  end

%   if (burnin/2+forgetting >= mcmc_iter)
%       error('FORGETTING is set too large.')
%   end
   
% WE ARE NOW READY TO UPDATE THE PROPOSAL DISTRIBUTION
%     mcmc_iter
   if (mcmc_iter == burnin) || (mcmc_iter == lastCovupdate+1) 
         if verbose
            fprintf('\nMCMC iter %d: proposal kernel updated...',mcmc_iter)
         end
         MCMC_iter = MCMC(1:mcmc_iter,:);
         save('MCMC_iter','MCMC_iter')
         mean_theta = mean(accepted_thetasimsum(burnin/2+forgetting:mcmc_iter,1:dtheta));
         mean_simsum = mean(accepted_thetasimsum(burnin/2+forgetting:mcmc_iter,dtheta+1:end));
         all_cov_thetasimsum = cov(accepted_thetasimsum(burnin/2+forgetting:mcmc_iter,:));
         cov_theta = all_cov_thetasimsum(1:dtheta,1:dtheta);
         cov_simsum = all_cov_thetasimsum(dtheta+1:end,dtheta+1:end);
         cov_thetasimsum = all_cov_thetasimsum(1:dtheta,dtheta+1:end);
         cov_simsumtheta = cov_thetasimsum';
         proposal_mean = mean_theta' + cov_thetasimsum * (cov_simsum \(sobs-mean_simsum'));
         proposal_mean =  proposal_mean';  % must be a row vector when passed to mvnrnd()
         proposal_cov = cov_theta - cov_thetasimsum * (cov_simsum \ cov_simsumtheta);
         lastCovupdate = mcmc_iter;
         [~, notposdef] = cholcov(proposal_cov);
         if isnan(notposdef) || notposdef>0
             % trick
             proposal_cov = (proposal_cov + proposal_cov.') / 2;
             [~, notposdef] = cholcov(proposal_cov);
             if isnan(notposdef) || notposdef>0
                [L, DMC, P] = modchol_ldlt(proposal_cov);
              %  perturbation = P'*L*DMC*L'*P - proposal_cov;
                proposal_cov = P'*L*DMC*L'*P;
             %   proposal_cov = nearestSPD(proposal_cov);
             end
         end
    end
end

    
end