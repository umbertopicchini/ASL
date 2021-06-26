function MCMC= bslmcmc(thetastart,data,numsim,R_mcmc,cov_current,burnin,length_CoVupdate,mcwm)
% thetastart: starting parameter values
% numsim: number of simulated datasets at teach mcmc iteration
% R_mcmc: number of mcmc iterations
% standard deviations for the diagonal covariance matrix of MRW

nobs = length(data);
MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

sobs = gk_sumstat(data);     % summary statistics of data

%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::


loglik_old = gk_synlik(theta_old,sobs,nobs,numsim);

%:::::::::::::::::::::::::::::: PRIORS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
% evaluate priors at old parameters
prior_old =  gk_prior(theta_old);
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


if isinf(loglik_old) || (prior_old==0) || isnan(loglik_old)
  loglik_old = -1e300;
  warning("The initial proposal is not admissible. We assign a loglikelihood = -1e300...")
end

% propose a value for parameters using Gaussian random walk
theta = mvnrnd(theta_old,cov_current);


loglik = gk_synlik(theta,sobs,nobs,numsim);

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
else
  % reject proposal
  MCMC(2,:) = theta_old;
end



for mcmc_iter = 3:burnin/2 

   theta = mvnrnd(theta_old,cov_current);
   loglik = gk_synlik(theta,sobs,nobs,numsim);
   if mcwm
       loglik_old = gk_synlik(theta_old,sobs,nobs,numsim);
   end

   % evaluate priors at proposed parameters
   prior =  gk_prior(theta);

  if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
     MCMC(mcmc_iter,:) = theta;
     loglik_old = loglik;
     theta_old = theta;
     prior_old = prior;
  else
     MCMC(mcmc_iter,:) = theta_old;
  end
end


lastCovupdate = 0;
accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values

for mcmc_iter = burnin/2+1:R_mcmc  

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
    
    
   loglik = gk_synlik(theta,sobs,nobs,numsim);
   if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
       if mcmc_iter <= burnin
          loglik_old = gk_synlik(theta_old,sobs,nobs,numsim);
       end
   end

   % evaluate priors at proposed parameters
   prior =  gk_prior(theta);


if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
   MCMC(mcmc_iter,:) = theta;
   loglik_old = loglik;
   theta_old = theta;
   prior_old = prior;
   accept_proposal=accept_proposal+1;
  else
     MCMC(mcmc_iter,:) = theta_old;
end 

end


end
