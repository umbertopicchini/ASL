function [loglik,mean_simsummaries,cov_simsummaries,simsummaries] = recruitment_synlik(bigtheta,sobs,numsim)

  simsummaries = recruitment_simsummaries(bigtheta,numsim);  % summary statistics of data
  dsum = length(sobs);

  mean_simsummaries = mean(simsummaries,2,'omitnan');
  cov_simsummaries = cov(simsummaries','omitrows');

M = (numsim-1)*cov_simsummaries;
[~,positive] = chol(M);
if positive>0  
    try
      M = (M+M.')/2;  
      M = nearestSPD(M);
    catch
      try 
          M = (M+M.')/2;   
          M = nearestSPD(M+1e-8*eye(dsum));
      catch
          loglik = -inf;
          return
      end
    end
end

% unbiased estimator for a Gaussian density (see Price and Drovandi 2016)
phi_argument = M - (sobs-mean_simsummaries) * (sobs-mean_simsummaries)'/(1-1/numsim);
[~,positive] = chol(phi_argument);
if positive>0  
    phi_argument = nearestSPD(phi_argument);
end
[~,positive] = chol(phi_argument);
if positive>0  
   loglik = -inf
   return
end


loglik = -(numsim-dsum-2)/2 * logdet(M,'chol') + ((numsim-dsum-3)/2) * logdet(phi_argument,'chol') ;

end
