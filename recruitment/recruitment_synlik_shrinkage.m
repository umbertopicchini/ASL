function [loglik,mean_simsummaries,cov_simsummaries,simsummaries] = recruitment_synlik(bigtheta,sobs,n,nbin,numsim,unif_draws,shrinkage,gamma)

  simsummaries = recruitment_simsummaries(bigtheta,n,nbin,numsim,unif_draws);   % summary statistics of data
  dsum = length(sobs);

  mean_simsummaries = mean(simsummaries,2);
  cov_simsummaries = cov(simsummaries');

if shrinkage  % see Warton (2008) Penalized Normal Likelihood and Ridge Regularizationof Correlation and Covariance Matrices, Journal of the American Statistical Association, 103:481,340-349, DOI: 10.1198/016214508000000021
    cov_diag = diag(cov_simsummaries);
    D_sqrt_inv = diag(cov_diag.^(-0.5));
    R = D_sqrt_inv * cov_simsummaries * D_sqrt_inv; % sample correlation matrix
    D_sqrt = diag(cov_diag.^(0.5));
    cov_simsummaries = D_sqrt * (gamma*R+(1-gamma)*eye(length(sobs))) * D_sqrt;
end



M = (numsim-1)*cov_simsummaries;
if ( (sum(any(isnan(M)))) || (sum(any(isinf(M)))) )
   loglik = -inf
   return
end

[~,positive] = chol(M);
if positive>0  
    M = nearestSPD(M);
   %[L, DMC, P] = modchol_ldlt(M);
   % M= P'*L*DMC*L'*P;
end

% unbiased estimator for a Gaussian density (see Price and Drovandi 2016)
phi_argument = M - (sobs-mean_simsummaries) * (sobs-mean_simsummaries)'/(1-1/numsim);
[~,positive] = chol(phi_argument);
if positive>0  
    phi_argument = nearestSPD(phi_argument);
  % [L, DMC, P] = modchol_ldlt(phi_argument);
  %  phi_argument = P'*L*DMC*L'*P;
end
[~,positive] = chol(phi_argument);
if positive>0  
   loglik = -inf
   return
end


loglik = -(numsim-dsum-2)/2 * logdet(M,'chol') + ((numsim-dsum-3)/2) * logdet(phi_argument,'chol') ;

end
