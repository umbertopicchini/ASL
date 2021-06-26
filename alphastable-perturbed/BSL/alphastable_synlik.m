function [loglik,mean_simsummaries,cov_simsummaries,simsummaries] = alphastable_synlik(thetatilde,sobs,covariates,nobs,numsim)

%:::: WARNING: here we are plugging the TRUE value of gamma according to
%this specific case study. 
gamma_true = covariates;
% see the comment on page 3746 of Peters, G. W., Sisson, S. A., & Fan, Y. (2012). Likelihood-free Bayesian inference for α-stable models. Computational Statistics & Data Analysis, 56(11), 3743-3756.
%:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


% transformation as in Ong et al https://doi.org/10.1007/s11222-017-9773-3
alphatilde = thetatilde(1);
betatilde  = thetatilde(2);
gammatilde = thetatilde(3);
deltatilde = thetatilde(4);

% transform back to the original parameters
alpha = (0.5 + 2*exp(alphatilde)) / (1+exp(alphatilde));
beta = (exp(betatilde)-1)/(1+exp(betatilde));
gamma = exp(gammatilde);
delta = deltatilde;

% generate from a *perturbed* alpha-stable model
% we call it "perturbed" since the lines below consider the case where
% alpha \in (0.97,1.03) and the case where alpha is outside such interval,
% whereas a regular alpha-stable would consider the case where alpha==1 and
% the case alpha != 1
W = exprnd(1,nobs,numsim);
U = unifrnd(-pi/2,pi/2,nobs,numsim);

if alpha>=0.97 && alpha<= 1.03
    ybar = 2/pi * ((pi/2+beta*U).*tan(U) -beta * log( (W.*cos(U))./(pi/2+beta*U) ));
    simdata = gamma*ybar + (2/pi) * beta*gamma * log(gamma) + delta;
else
    S = (1+beta^2* (tan(pi*alpha/2))^2)^(1/(2*alpha));
    B = 1/ alpha * atan(beta*tan(pi*alpha/2));
    ybar = S * (sin((alpha)*(U+B))) ./ ((cos(U)).^(1/alpha)) .* ( cos(U-alpha*(U+B)) ./ W).^((1-alpha)/alpha) ;
    simdata = gamma*ybar + delta;
end


simsummaries = alphastable_sumstat(simdata,gamma_true);   % summary statistics of data

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
