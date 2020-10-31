function [loglik,cov_simsummaries,simsummaries] = recruitment_robustsynlik(bigtheta,sobs,numsim)

% returns the log-synlik as from An et al 2020.
% Z. An, D. J. Nott, and C. Drovandi.  Robust bayesian synthetic likelihood via a semi-parametricapproach.Statistics and Computing, 30:543–557, 2020.

  simsummaries = recruitment_simsummaries(bigtheta,numsim);   % summary statistics of data
  dsum = length(sobs);

 % mean_simsummaries = mean(simsummaries,2);
  cov_simsummaries = cov(simsummaries');

logpdf_sobs = zeros(dsum,1);
cdf_sobs = zeros(dsum,1);
eta_sobs = zeros(dsum,1);
ranks = zeros(dsum,numsim);

  for jj=1:dsum
     % pdf_sobs(jj) = ksdensity(simsummaries(jj,:),sobs(jj),'function','pdf','Kernel','normal');
      logpdf_sobs(jj) = loggausskernel(simsummaries(jj,:),sobs(jj));
      try
         cdf_sobs(jj) = ksdensity(simsummaries(jj,:),sobs(jj),'function','cdf','Kernel','normal');
      catch
          loglik = -inf;
          return
      end
      if cdf_sobs(jj)==1
         cdf_sobs(jj) = 0.999999999999999; % or otherwise it will give eta_sobs(jj) = inf in next line
      end
      eta_sobs(jj) = norminv(cdf_sobs(jj));
  end

  for jj=1:dsum
      ranks(jj,:) = tiedrank(simsummaries(jj,:));
  end


num_rankgauss = zeros(dsum,dsum);

for jj=1:dsum
    for jj2=1:dsum
       num_rankgauss(jj,jj2) = sum(norminv(ranks(jj,:)/(numsim+1)).*norminv(ranks(jj2,:)/(numsim+1)));
    end
end

den_rankgauss = sum((norminv([1:numsim]./(numsim+1))).^2);
gauss_rank = num_rankgauss./den_rankgauss;

[~,positive] = chol(gauss_rank);
if positive>0  
   loglik = -inf;
   return
end


loglik = -1/2 * logdet(gauss_rank,'chol') -1/2 * eta_sobs'*(inv(gauss_rank)-eye(dsum))*eta_sobs + sum(logpdf_sobs);

    function out = loggausskernel(xi,x)
        
        h = numsim^(-0.2) * 0.9 * min(std(xi),iqr(xi)/1.349); % plug-in bandwidth
        % the commented line below would compute a Gaussian kernel smoothed pdf
      %  out = 1/(numsim*h) * 1/sqrt(2*pi) * sum(exp(-0.5/h^2 * (x-xi).^2));
        z = -0.5/h^2 * (x-xi).^2;
        % here is the log of the kernel smoothed pdf
        out = -log(numsim*h) + logsumexp(z);
    end

end
