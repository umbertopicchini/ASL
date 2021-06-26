function MCMC= bslmcmc(bigthetastart,parmask,parbase,sobs,nobs,nbin,numsim,R_mcmc,cov_current,burnin,length_CoVupdate, shrinkage, gamma)

% thetastart: starting parameter values
% numsim: number of simulated datasets at teach mcmc iteration
% R_mcmc: number of mcmc iterations
% standard deviations for the diagonal covariance matrix of MRW

if mod(numsim,2)>0
    error('NUMSIM should be an even integer. Also, it should be a multiple of NUMGROUPS')
end

thetastart = param_mask(bigthetastart,parmask);

MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

bigtheta_old = param_unmask(theta_old,parmask,parbase);

%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::

unif = rand(nobs,numsim);
if shrinkage
    [loglik_old,~,~,simsum] = astroSL_synlik_shrinkage(bigtheta_old,sobs,nobs,nbin,numsim,unif, shrinkage, gamma);
else
    [loglik_old,~,~,simsum] = astroSL_synlik(bigtheta_old,sobs,nobs,nbin,numsim,unif);
end



%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at old parameters
prior_old =  astroSL_prior(theta_old);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
unif = rand(nobs,numsim);  
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
else
  % reject proposal
  MCMC(2,:) = theta_old;
end

accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values

for mcmc_iter = 3:R_mcmc
    
    %::::::::: ADAPTATION OF THE COVARIANCE MATRIX FOR THE PARAMETERS PROPOSAL :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    %::::::::: here we follow the adaptive Metropolis method as in:
    %::::::::  Haario et al. (2001) "An adaptive Metropolis algorithm", Bernoulli Volume 7, 223-242.
      if mcmc_iter == burnin
             if burnin >= length_CoVupdate
                lastCovupdate = burnin-length_CoVupdate;
             end
             cov_last = cov_current;
       end
       if (mcmc_iter < burnin)
          theta = mvnrnd(theta_old,cov_current);
       else
           if (mcmc_iter == lastCovupdate+length_CoVupdate) 
               covupdate = cov(MCMC(burnin/2:mcmc_iter-1,1:end));
               % compute equation (1) in Haario et al.
               cov_current = (2.38^2)/length(theta)*covupdate +  (2.38^2)/length(theta) * 1e-8 * eye(length(theta)) ;
               theta = mvnrnd(theta_old,cov_current);
               cov_last = cov_current;
               lastCovupdate = mcmc_iter;
               fprintf('\nMCMC iteration -- adapting covariance...')
               fprintf('\nMCMC iteration #%d -- acceptance ratio %4.3f percent',mcmc_iter,accept_proposal/num_proposal*100)
               accept_proposal=0;
               num_proposal=0;
               MCMC_temp = MCMC(1:mcmc_iter-1,:);
               save('THETAmatrix_temp','MCMC_temp');
           else
              % Here there is no "adaptation" for the covariance matrix,
              % hence we use the same one obtained at last update
                theta = mvnrnd(theta_old,cov_last);
           end
       end
   
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::   
   
   num_proposal = num_proposal+1; 
   bigtheta = param_unmask(theta,parmask,parbase);
   
   unif = rand(nobs,numsim);
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
   else
      MCMC(mcmc_iter,:) = theta_old;
   end 
    
end