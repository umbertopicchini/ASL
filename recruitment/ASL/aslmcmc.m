function [MCMC,proposal_cov]=  aslmcmc(thetastart,sobs,numsim,R_mcmc,step_rw,burnin,randomwalk_iterstart,verbose,robust,mcwm,length_CoVupdate)
% thetastart: starting parameter values
% numsim: number of simulated datasets at teach mcmc iteration
% R_mcmc: number of mcmc iterations
% standard deviations for the diagonal covariance matrix of MRW

if mod(numsim,2)>0
    error('NUMSIM should be an even integer.')
end

if randomwalk_iterstart <= burnin
    error('BURNIN must be smaller than randomwalk_iterstart')
end

MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

dtheta = length(thetastart);
dsobs = length(sobs);
%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::

if robust
   [loglik_old,cov_simsummaries,simsum] = recruitment_robustsynlik(theta_old,sobs,numsim); 
else
   [loglik_old,mean_simsummaries,cov_simsummaries,simsum] = recruitment_synlik(theta_old,sobs,numsim);
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
if isinf(loglik_old) || isnan(loglik_old)
  loglik_old = -1e300
  warning("The initial proposal is not admissible. We assign a loglikelihood = -1e300...")
end


% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);

if robust
   [loglik,cov_simsummaries,simsum] = recruitment_robustsynlik(theta,sobs,numsim); 
else
   [loglik,~,cov_simsummaries,simsum] = recruitment_synlik(theta,sobs,numsim);
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
  mean_accepted_simsum = mean(accepted_simsum,1);
  accepted_thetasimsum(2,:) = [theta_old,mean_accepted_simsum];
end

% we execute an MCMC for only burnin/2 iterations only in otder to compute
% covariances between theta and summary statistics. It is unsafe to compute
% also other quantities so early in the MCMC chain and instead wait  a little more, 
% until the chain reaches more promising areas 

for mcmc_iter = 3:burnin/2 

   theta = mvnrnd(theta_old,cov_current);
   
   if robust
      [loglik,cov_simsummaries,simsum] = recruitment_robustsynlik(theta,sobs,numsim); 
       if mcwm
          [loglik_old,~,~] = recruitment_robustsynlik(theta_old,sobs,numsim); 
       end
   else
      [loglik,~,cov_simsummaries,simsum] = recruitment_synlik(theta,sobs,numsim);
      if mcwm
          [loglik_old,~,~] = recruitment_synlik(theta_old,sobs,numsim); 
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
accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values

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
   
   num_proposal = num_proposal+1; 
   
   if robust
      [loglik,cov_simsummaries,simsum] = recruitment_robustsynlik(theta,sobs,numsim);
      if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
        if mcmc_iter <= burnin
           [loglik_old,~,~] = recruitment_robustsynlik(theta_old,sobs,numsim);
         end
      end
   else
      [loglik,~,cov_simsummaries,simsum] = recruitment_synlik(theta,sobs,numsim);
      if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
        if mcmc_iter <= burnin
           [loglik_old,~,~] = recruitment_synlik(theta_old,sobs,numsim);
         end
      end
   end

   % evaluate priors at proposed parameters
   prior =  recruitment_prior(theta);


   if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
      if verbose
         fprintf('\nMCMC iter %d: proposal accepted...',mcmc_iter)
      end
      accept_proposal=accept_proposal+1;
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
