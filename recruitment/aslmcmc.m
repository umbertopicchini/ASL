function [MCMC,proposal_cov]= aslmcmc(bigthetastart,parmask,parbase,sobs,nobs,numsim,R_mcmc,step_rw,burnin,forgetting,verbose,expandcov,robust,mcwm,lengthcovupdate)


if mod(numsim,2)>0
    error('NUMSIM should be an even integer. Also, it should be a multiple of NUMGROUPS')
end

thetastart = param_mask(bigthetastart,parmask);

MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

bigtheta_old = param_unmask(theta_old,parmask,parbase);

dtheta = length(thetastart);
dsobs = length(sobs);
%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::

if robust
   [loglik_old,cov_simsummaries,simsum] = recruitment_robustsynlik(bigtheta_old,sobs,numsim); 
else
   [loglik_old,mean_simsummaries,cov_simsummaries,simsum] = recruitment_synlik(bigtheta_old,sobs,numsim);
end

accepted_simsum = simsum';
accepted_thetasimsum = zeros(R_mcmc,dtheta+dsobs);

accepted_thetasimsum(1,:) = [theta_old,mean(accepted_simsum,1)];

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at old parameters
prior_old =  recruitment_prior(theta_old);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

% initial (diagonal) covariance matrix for the Gaussian proposal
cov_current = step_rw.^2 .* eye(length(thetastart));
bigtheta_old = param_unmask(theta_old,parmask,parbase);
if isinf(loglik_old) || (prior_old==0) || isnan(loglik_old)
  loglik_old = -1e300;
  warning("The initial proposal is not admissible. Assigning the loglikelihood a value -1e300...")
end


% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);
bigtheta = param_unmask(theta,parmask,parbase);

if robust
   [loglik,cov_simsummaries,simsum] = recruitment_robustsynlik(bigtheta,sobs,numsim); 
else
   [loglik,~,cov_simsummaries,simsum] = recruitment_synlik(bigtheta,sobs,numsim);
end



%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at proposed parameters
prior =  recruitment_prior(theta);
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
  accepted_thetasimsum(2,:) = [theta_old,mean(accepted_simsum,1)];
end

% we execute an MCMC for only burnin/2 iterations only in otder to compute
% covariances between theta and summary statistics. It is unsafe to compute
% also other quantities so early in the MCMC chain and instead wait  a little more, 
% until the chain reaches more promising areas 

for mcmc_iter = 3:burnin/2 

   theta = mvnrnd(theta_old,cov_current);
   bigtheta = param_unmask(theta,parmask,parbase);
   
   if robust
      [loglik,cov_simsummaries,simsum] = recruitment_robustsynlik(bigtheta,sobs,numsim); 
       if mcwm
           bigtheta_old = param_unmask(theta_old,parmask,parbase);
          [loglik_old,~,~] = recruitment_robustsynlik(bigtheta_old,sobs,numsim); 
       end
   else
      [loglik,~,cov_simsummaries,simsum] = recruitment_synlik(bigtheta,sobs,numsim);
      if mcwm
          bigtheta_old = param_unmask(theta_old,parmask,parbase);
          [loglik_old,~,~] = recruitment_synlik(bigtheta_old,sobs,numsim); 
       end
   end

   % evaluate priors at proposed parameters
   prior =  recruitment_prior(theta);
   
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

  % theta = mvnrnd(theta_old,cov_current);
   if mcmc_iter <= burnin
       theta = mvnrnd(theta_old,cov_current);  % Gaussian random walk
   else
       numproposal = numproposal+1;
     %  proposal_cov
       % here we update the expandcov factor according to the current
       % acceptance rate
       if ~stop_update_expandcov
         if (mcmc_iter == burnin + lengthcovupdate) || (mcmc_iter == lastexpandcovupdate+lengthcovupdate)
            if acceptrate > 0.20  % acceptance rate is currently > 20 percent
               expandcov = expandcov + 0.25*expandcov;  % increase expandcov by 25 percent
             elseif acceptrate < 0.15 % acceptance rate is currently < 15 percent
                expandcov = expandcov - 0.05*expandcov; % decrease expandcov by 5 percent
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
       % Student's independence sampler
%        dist =  (sobs-mean_simsum')'*((cov_simsum)\(sobs-mean_simsum'));
%        Z = mvnrnd(zeros(dtheta,1),eye(dtheta))';
%        nu = 5;
%        theta = proposal_mean + ((sqrt((nu+dist)/(nu+dsobs))*sqrtm(expandcov * proposal_cov)) * (Z/sqrt(chi2rnd(nu+dsobs)/(nu+dsobs))))'  % Student's independence sampler
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
%           % proposal_cov
%        end
%       *************
%       % theta = mvnrnd(proposal_mean,expandcov*proposal_cov + expandcov*1e-8*eye(length(thetastart)));  % Gaussian independence sampler
%        dist =  (sobs-mean_simsum')'*((cov_simsum)\(sobs-mean_simsum'));
%        Z = mvnrnd(zeros(dtheta,1),eye(dtheta))';
%        nu = 5;
%        theta = proposal_mean + ((sqrt((nu+dist)/(nu+dsobs))*sqrtm(proposal_cov)) * (Z/sqrt(chi2rnd(nu+dsobs)/(nu+dsobs))))'  % Student's independence sampler
   end
   
   if ~isreal(theta)
       MCMC(mcmc_iter,:) = theta_old;
       sampled_indeces = randsample(numsim,numsim,true);
       accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean(accepted_simsum(sampled_indeces,:),1)];
       continue
   end
   
   bigtheta = param_unmask(theta,parmask,parbase);
   
   if robust
      [loglik,cov_simsummaries,simsum] = recruitment_robustsynlik(bigtheta,sobs,numsim);
      if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
        if mcmc_iter <= burnin
           bigtheta_old = param_unmask(theta_old,parmask,parbase);
           [loglik_old,~,~] = recruitment_robustsynlik(bigtheta_old,sobs,numsim);
         end
      end
   else
      [loglik,~,cov_simsummaries,simsum] = recruitment_synlik(bigtheta,sobs,numsim);
      if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
        if mcmc_iter <= burnin
           bigtheta_old = param_unmask(theta_old,parmask,parbase);
           [loglik_old,~,~] = recruitment_synlik(bigtheta_old,sobs,numsim);
         end
      end
   end


   % evaluate priors at proposed parameters
   prior =  recruitment_prior(theta);


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
      if verbose
         fprintf('\nMCMC iter %d: acceptance...',mcmc_iter)
      end
   else
      MCMC(mcmc_iter,:) = theta_old;
      sampled_indeces = randsample(numsim,numsim,true);
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,mean(accepted_simsum(sampled_indeces,:),1)];
   end 
   
  if (forgetting > 0) && (mcmc_iter > burnin) 
     forgetting = forgetting +1;  
  end

   
% WE ARE NOW READY TO UPDATE THE PROPOSAL DISTRIBUTION VIA ASL
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
                proposal_cov = P'*L*DMC*L'*P;
             %   proposal_cov = nearestSPD(proposal_cov);
             end
         end
    end
end

    
end