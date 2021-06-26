function MCMC= bslmcmc(thetastart,data,numsim,R_mcmc,cov_current,burnin,length_CoVupdate,numgroups)
% thetastart: starting parameter values
% numsim: number of simulated datasets at teach mcmc iteration
% R_mcmc: number of mcmc iterations
% standard deviations for the diagonal covariance matrix of MRW

if mod(numsim,2)>0
    error('NUMSIM should be an even integer. ALso, it shouod be a multiple of NUMGROUPS')
end
if mod(numsim,numgroups)>0
    error('NUMGROUPS should be an even integer such that NUMSIM is a multiple of NUMGROUPS.')
end
if numgroups >= numsim
    error('NUMGROUPS should be smaller than NUMSIM.')
end

% prepare columns of indeces, useful for the correlated blocking strategy
indeces = [1:numsim]; % all integer indeces 1,2,...,numsim
blocks = reshape(indeces, numsim/numgroups, numgroups); % block is now a [numsim/numgroups] x numgroups matrix where all the indeces created above are arranged into numgroups columns


nobs = length(data);
MCMC = zeros(R_mcmc,length(thetastart));
MCMC(1,:) = thetastart;
theta_old = thetastart;

sobs = gk_sumstat(data);     % summary statistics of data

%:::::::::::::::::::::::::::::: INITIALIZATION  ::::::::::::::::

z_gauss_old = randn(nobs,numsim);
loglik_old = gk_synlik(theta_old,sobs,numsim,z_gauss_old);

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

select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
z_gauss = z_gauss_old(:);  % make the matrix of random draws a vector to simplify things
z_gauss(select_indeces) = randn(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same 
z_gauss = reshape(z_gauss,nobs,numsim); % go back to original dimensions
loglik = gk_synlik(theta,sobs,numsim,z_gauss);

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
  z_gauss_old = z_gauss;
else
  % reject proposal
  MCMC(2,:) = theta_old;
end



for mcmc_iter = 3:burnin/2 

   theta = mvnrnd(theta_old,cov_current);
   
   select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
   select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
   z_gauss = z_gauss_old(:);  % make the matrix of random draws a vector to simplify things
   z_gauss(select_indeces) = randn(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same 
   z_gauss = reshape(z_gauss,nobs,numsim); % go back to original dimensions
   loglik = gk_synlik(theta,sobs,numsim,z_gauss);
   
   % evaluate priors at proposed parameters
   prior =  gk_prior(theta);

  if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
     MCMC(mcmc_iter,:) = theta;
     loglik_old = loglik;
     theta_old = theta;
     prior_old = prior;
     z_gauss_old = z_gauss;
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
    
   % update pseudo-random numbers in a randomly selected block of indeces
   select_block = randi(numgroups); % select a block randomly from the set {1,2...,numgroups} with constant probability 1/numgroups
   select_indeces = blocks(:,select_block); % select a set of indeces corresponding to the random block we just obtained
   z_gauss = z_gauss_old(:);  % make the matrix of random draws a vector to simplify things
   z_gauss(select_indeces) = randn(numsim/numgroups,1); % put new random numbers in the selected block of indeces, while remaining random numbers stay the same as in the previous iteration
   z_gauss = reshape(z_gauss,nobs,numsim); % go back to original dimensions
   loglik = gk_synlik(theta,sobs,numsim,z_gauss);

   % evaluate priors at proposed parameters
   prior =  gk_prior(theta);


if (log(rand) < loglik-loglik_old +log(prior)-log(prior_old))
   MCMC(mcmc_iter,:) = theta;
   loglik_old = loglik;
   theta_old = theta;
   prior_old = prior;
   z_gauss_old = z_gauss;
   accept_proposal=accept_proposal+1;
  else
     MCMC(mcmc_iter,:) = theta_old;
end 

end


end
