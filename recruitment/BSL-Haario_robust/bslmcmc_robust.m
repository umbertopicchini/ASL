function MCMC = bslmcmc(bigthetastart,parmask,parbase,sobs,nobs,numsim,R_mcmc,step_rw,burnin,length_CoVupdate,robust,mcwm)
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

if robust
   [loglik_old,cov_simsummaries,simsum] = recruitment_robustsynlik(bigtheta_old,sobs,numsim); 
else
   [loglik_old,~,cov_simsummaries,simsum] = recruitment_synlik(bigtheta_old,sobs,numsim);
end

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
else
  % reject proposal
  MCMC(2,:) = theta_old;
end

accept_proposal=0;  % start the counter for the number of accepted proposals
num_proposal=0;     % start the counter for the total number of proposed values


for mcmc_iter = 2:R_mcmc
mcmc_iter
      if mcmc_iter == burnin
             if burnin >= length_CoVupdate
                lastCovupdate = burnin-length_CoVupdate;
             end
             cov_last = cov_current;
       end
       if (mcmc_iter < burnin)
          cov_current = diag(step_rw.^2); 
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
   
   
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    %::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::   

   bigtheta = param_unmask(theta,parmask,parbase);
   
   if robust
      [loglik,~,~] = recruitment_robustsynlik(bigtheta,sobs,numsim); 
      if mcwm  % "refresh" the old likelihood (perform Markov chain Within Metropolis)
        if mcmc_iter <= burnin
           bigtheta_old = param_unmask(theta_old,parmask,parbase);
           [loglik_old,~,~] = recruitment_robustsynlik(bigtheta_old,sobs,numsim);
         end
      end
   else
      [loglik,~,~,~] = recruitment_synlik(bigtheta,sobs,numsim);
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
      accept_proposal=accept_proposal+1;
      MCMC(mcmc_iter,:) = theta;
      loglik_old = loglik;
      theta_old = theta;
      prior_old = prior;
   else
      MCMC(mcmc_iter,:) = theta_old;
   end 
    
end