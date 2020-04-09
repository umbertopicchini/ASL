function [MCMC,proposal_cov]= bslmcmc(bigthetastart,parmask,parbase,sobs,nobs,nbin,numsim,R_mcmc,step_rw,burnin,numgroups, shrinkage, gamma,forgetting,verbose,expandcov,haario_iter,length_CoVupdate)


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

MCMC = zeros(R_mcmc+haario_iter,length(thetastart));
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
accepted_thetasimsum(1,:) = [theta_old,accepted_simsum(1,:)];

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
end


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
  accepted_thetasimsum(2,:) = [theta_old,accepted_simsum(randi(numsim),:)];
else
  % reject proposal
  MCMC(2,:) = theta_old;
  accepted_thetasimsum(2,:) = [theta_old,accepted_simsum(randi(numsim),:)];
end

% we execute an MCMC for only burnin/2 iterations only in order to compute
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
     accepted_thetasimsum(mcmc_iter,:) = [theta_old,accepted_simsum(randi(numsim),:)];
     if verbose
        fprintf('\nMCMC iter %d: acceptance...',mcmc_iter)
     end
  else
     MCMC(mcmc_iter,:) = theta_old;
     accepted_thetasimsum(mcmc_iter,:) = [theta_old,accepted_simsum(randi(numsim),:)];
  end
end

lastCovupdate = 0;

for mcmc_iter = burnin/2+1:R_mcmc

   if mcmc_iter <= burnin
       theta = mvnrnd(theta_old,cov_current);  % Gaussian random walk
   else
        theta = mvnrnd(proposal_mean,expandcov*proposal_cov);  % Gaussian independence sampler
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
      MCMC(mcmc_iter,:) = theta;
      loglik_old = loglik;
      theta_old = theta;
      prior_old = prior;
      unif_old = unif;
      accepted_simsum = simsum';
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,accepted_simsum(randi(numsim),:)];
      if verbose
         fprintf('\nMCMC iter %d: acceptance...',mcmc_iter)
      end
   else
      MCMC(mcmc_iter,:) = theta_old;
      accepted_thetasimsum(mcmc_iter,:) = [theta_old,accepted_simsum(randi(numsim),:)];
   end 
   
  if (forgetting > 0) && (mcmc_iter > burnin) 
     forgetting = forgetting +1;  
  end

   
% WE ARE NOW READY TO UPDATE THE PROPOSAL DISTRIBUTION USING ASL
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

lastCovupdate = R_mcmc;
accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values

for mcmc_iter = R_mcmc+1:R_mcmc+haario_iter
    
    %::::::::: ADAPTATION OF THE COVARIANCE MATRIX FOR THE PARAMETERS PROPOSAL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    %::::::::: here we follow the adaptive Metropolis method as in:
    %::::::::  Haario et al. (2001) "An adaptive Metropolis algorithm", Bernoulli Volume 7, 223-242.
       if mcmc_iter == R_mcmc+1
             cov_last = expandcov*proposal_cov;
             cov_old = cov_last;
             sum_old = sum(MCMC(mcmc_iter-length_CoVupdate:mcmc_iter-1,:));
             covupdate = cov_update(theta_old,sum_old,size(MCMC(mcmc_iter-length_CoVupdate:mcmc_iter-1,:),1),cov_old);
             cov_last = covupdate;
             lastCovupdate = mcmc_iter;
       end
       if (mcmc_iter == lastCovupdate+length_CoVupdate) 
               fprintf('\nMCMC iteration -- adapting covariance...')
               fprintf('\nMCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
               % we do not need to recompute the covariance on the whole
               % past history in a brutal way: we can use a recursive
               % formula. See the reference in the file cov_update.m
               covupdate = cov_update(MCMC(lastCovupdate+1:mcmc_iter-1,:),sum_old,length_CoVupdate-1,cov_old);
               sum_old = sum(MCMC(lastCovupdate+1:mcmc_iter-1,:));
               cov_old = cov(MCMC(lastCovupdate+1:mcmc_iter-1,:));
               % compute equation (1) in Haario et al.
               cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-8 * eye(length(theta)); 
               theta = mvnrnd(theta_old,cov_current);
               cov_last = cov_current;
               lastCovupdate = mcmc_iter;
               accept_proposal=0;
               num_proposal=0;
               MCMC_temp = MCMC(1:mcmc_iter-1,:);
               save('THETAmatrix_temp','MCMC_temp');
           else
              % Here there is no "adaptation" for the covariance matrix,
              % hence we use the same one obtained at last update
                theta = mvnrnd(theta_old,cov_last);
       end
   
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::   
   
   num_proposal = num_proposal+1; 
   bigtheta = param_unmask(theta,parmask,parbase);
   
   % update pseudo-random numbers in a randomly selected block of indeces
   select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
   select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
   unif = unif_old(:);  % make the matrix of random draws a vector to simplify things
   unif(select_indeces) = rand(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same as in the previous iteration
   unif = reshape(unif,nobs,numsim); % go back to original dimensions
   if shrinkage
      [loglik,~,~,~] = astroSL_synlik_shrinkage(bigtheta,sobs,nobs,nbin,numsim,unif, shrinkage, gamma);
   else
      [loglik,~,~,~] = astroSL_synlik(bigtheta,sobs,nobs,nbin,numsim,unif);
   end

   % evaluate priors at proposed parameters
   prior =  astroSL_prior(theta);


   if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
      accept_proposal=accept_proposal+1;
      MCMC(mcmc_iter,:) = theta;
      loglik_old = loglik;
      theta_old = theta;
      prior_old = prior;
      unif_old = unif;
   else
      MCMC(mcmc_iter,:) = theta_old;
   end 
    
end