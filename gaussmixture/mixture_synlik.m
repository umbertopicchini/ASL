function [loglik,mean_simsummaries,cov_simsummaries,simsummaries] = mixture_synlik(theta,sobs,nobs,numsim)
  
%  theta = sort(theta);
  theta_dimension_1 = sort([theta(1),theta(3)]);
  theta_dimension_2 = sort([theta(2),theta(4)]);
  
 % theta = sort(theta);
  
%   m1 = theta_dimension_1(1);
%   m2 = theta_dimension_2(1);
%   m3 = theta_dimension_1(2);
%   m4 = theta_dimension_2(2);

%   mu1 = theta(1:2);
%   mu2 = theta(3:4);
%   mu = [mu1;mu2];

  mu = [theta_dimension_1(1), theta_dimension_2(1); theta_dimension_1(2), theta_dimension_2(2)];
  
  % this is the ground truth covariance
sigma1 = [4^2 0; 0 4^2];
sigma2 = [4^2 0; 0 4^2];
sigma(:,:,1) = sigma1;
sigma(:,:,2) = sigma2;
  
  simsummaries = zeros(4,numsim);
  
  prop = [1/2, 1/2];
  gm = gmdistribution(mu,sigma,prop); % 2-components Gaussian mixture with mixing proportions prop
  % generate numsim datasets and compute summaries for each dataset
  
  % this is done in parallel. Fell free to use 'for' instead of 'parfor' if
  % you don't have the Parallel Toolbox
  parfor ii=1:numsim % can also be 'for ii=1:numsim'
      simdata = random(gm,nobs);
      simsummaries(:,ii) = mixture_sumstat(simdata);
  end
  
  dsum = length(sobs);
  %simsummaries
  mean_simsummaries = mean(simsummaries,2);
  cov_simsummaries = cov(simsummaries');

M = (numsim-1)*cov_simsummaries;
[~,positive] = chol(M);
if positive>0  
    try
      M = (M+M.')/2;  
     % M = nearestSPD(M);
     [L, DMC, P] = modchol_ldlt(M);
      M = P'*L*DMC*L'*P;
    catch
      try 
          M = (M+M.')/2;   
        %  M = nearestSPD(M+1e-8*eye(dsum));
          [L, DMC, P] = modchol_ldlt(M+1e-8*eye(dsum));
          M = P'*L*DMC*L'*P;
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
    try
      phi_argument = (phi_argument+phi_argument.')/2;  
     % M = nearestSPD(M);
     [L, DMC, P] = modchol_ldlt(phi_argument);
      phi_argument = P'*L*DMC*L'*P;
    catch
      try 
          phi_argument = (phi_argument+phi_argument.')/2;   
        %  M = nearestSPD(M+1e-8*eye(dsum));
          [L, DMC, P] = modchol_ldlt(phi_argument+1e-8*eye(dsum));
          phi_argument = P'*L*DMC*L'*P;
      catch
          loglik = -inf;
          return
      end
    end
end

% [~, notposdef] = cholcov(phi_argument);
% if isnan(notposdef)
%    phi_argument = nearestSPD(phi_argument);
%    [~, notposdef] = cholcov(phi_argument);
%    if isnan(notposdef)
%        loglik = -inf;
%        return
%    end
% end


loglik = -(numsim-dsum-2)/2 * logdet(M,'chol') + ((numsim-dsum-3)/2) * logdet(phi_argument,'chol') ;

end
