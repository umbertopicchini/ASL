function loglik = gk_synlik(theta,sobs,numsim,z_gauss)

  simdata = gk_rnd(theta,z_gauss);  % the simulated data
  simsummaries = gk_sumstat(simdata);   % summary statistics of data
  dsum = length(sobs);

  mean_simsummaries = mean(simsummaries,2);
  cov_simsummaries = cov(simsummaries');

M = (numsim-1)*cov_simsummaries;

% unbiased estimator for a Gaussian density (see Price and Drovandi 2016)
phi_argument = M - (sobs-mean_simsummaries) * (sobs-mean_simsummaries)'/(1-1/numsim);
[~,positive] = chol(phi_argument);
if positive>0  
   loglik = -inf;
   return
end


loglik = -(numsim-dsum-2)/2 * logdet(M,'chol') + ((numsim-dsum-3)/2) * logdet(phi_argument,'chol') ;

end
