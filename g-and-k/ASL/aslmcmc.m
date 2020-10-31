function [MCMC,proposal_cov]= aslmcmc(thetastart,data,numsim,R_mcmc,step_rw,burnin,verbose,mcwm,expandcov)
% thetastart: starting parameter values
% numsim: number of simulated datasets at teach mcmc iteration
% R_mcmc: number of mcmc iterations
% standard deviations for the diagonal covariance matrix of MRW


nobs = length(data);
MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

sobs = gk_sumstat(data);     % summary statistics of data
dtheta = length(thetastart);
dsobs = length(sobs);
%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::

[loglik_old,~,cov_simsummaries,simsum] = gk_synlik(theta_old,sobs,nobs,numsim);
accepted_simsum = simsum';
accepted_thetasimsum = zeros(R_mcmc,dtheta+dsobs);

accepted_thetasimsum(1,:) = [theta_old,mean(accepted_simsum,1)];

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at old parameters
prior_old =  gk_prior(theta_old);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

% initial (diagonal) covariance matrix for the Gaussian proposal
cov_current = step_rw.^2 .* eye(length(thetastart));
if isinf(loglik_old) || isnan(loglik_old)
  loglik_old = -1e300
  warning("The initial proposal is not admissible. We assign a loglikelihood = -1e300...")
end


% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);

[loglik,~,cov_simsummaries,simsum] = gk_synlik(theta,sobs,nobs,numsim);

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at proposed parameters
prior =  gk_prior(theta);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

if log(rand) < loglik-loglik_old +log(prior)-log(prior_old)
  % here we accept our proposal theta
  MCMC(2,:) = theta;
  loglik_old = loglik;
  theta_old = theta;
  prior_old = prior;
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
   [loglik,~,cov_simsummaries,simsum] = gk_synlik(theta,sobs,nobs,numsim);
   if mcwm
       [loglik_old,~,~,~] = gk_synlik(theta_old,sobs,nobs,numsim);
   end

   % evaluate priors at proposed parameters
   prior =  gk_prior(theta);

  if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
     MCMC(mcmc_iter,:) = theta;
     loglik_old = loglik;
     theta_old = theta;
     prior_old = prior;
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

    

   if mcmc_iter <= burnin
       theta = mvnrnd(theta_old,cov_current);  % Gaussian random walk
   else
       numproposal = numproposal+1;
       % here we update the expandcov factor according to the current
       % acceptance rate
       if ~stop_update_expandcov
         if (mcmc_iter == burnin + 50) || (mcmc_iter == lastexpandcovupdate+50)
            if acceptrate > 0.20  % acceptance rate is currently > 20 percent
               expandcov = expandcov + 0.25*expandcov;  % increase expandcov by 25 percent
            elseif acceptrate < 0.15  % acceptance rate is currently < 15 percent
               expandcov = expandcov - 0.05*expandcov;  % decrease expandcov by 5 percent
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
   end
   
   [loglik,~,cov_simsummaries,simsum] = gk_synlik(theta,sobs,nobs,numsim);
   if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
       if mcmc_iter <= burnin
          [loglik_old,~,~,~] = gk_synlik(theta_old,sobs,nobs,numsim);
       end
   end

   % evaluate priors at proposed parameters
   prior =  gk_prior(theta);


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
      accepted_simsum = simsum';
      mean_accepted_simsum = mean(accepted_simsum,1);
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean_accepted_simsum];
   else
      MCMC(mcmc_iter,:) = theta_old;
      sampled_indeces = randsample(numsim,numsim,true);
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean(accepted_simsum(sampled_indeces,:),1)];
   end 
   
   
% WE ARE NOW READY TO UPDATE THE PROPOSAL DISTRIBUTION
%     mcmc_iter
   if (mcmc_iter == burnin) || (mcmc_iter == lastCovupdate+1) 
         if verbose
            fprintf('\nMCMC iter %d: proposal kernel updated...',mcmc_iter)
         end
         MCMC_iter = MCMC(1:mcmc_iter,:);
         save('MCMC_iter','MCMC_iter')
         mean_theta = mean(accepted_thetasimsum(burnin/2:mcmc_iter,1:dtheta));
         mean_simsum = mean(accepted_thetasimsum(burnin/2:mcmc_iter,dtheta+1:end));
         all_cov_thetasimsum = cov(accepted_thetasimsum(burnin/2:mcmc_iter,:));
         cov_theta = all_cov_thetasimsum(1:dtheta,1:dtheta);
         cov_simsum = all_cov_thetasimsum(dtheta+1:end,dtheta+1:end);
         cov_thetasimsum = all_cov_thetasimsum(1:dtheta,dtheta+1:end);
         cov_simsumtheta = cov_thetasimsum';
         proposal_mean = mean_theta' + cov_thetasimsum * (cov_simsum \(sobs-mean_simsum'));
         proposal_mean =  proposal_mean';  % must be a row vector when passed to mvnrnd()
         proposal_cov = cov_theta - cov_thetasimsum * (cov_simsum \ cov_simsumtheta);
         lastCovupdate = mcmc_iter;
         [~, notposdef] = cholcov(proposal_cov);
         if isnan(notposdef)
             proposal_cov = nearestSPD(proposal_cov);
         end
    end
end

end