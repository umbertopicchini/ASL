function out = recruitment_prior(theta)

% Returns the product of independent priors for parameters of a g-and-k distribution
% Input:  - theta, the vector of parameters to be estimated
% Output: - out, the product of the prior distributions set for each parameter.
%                possibily multiplied with a jacobian for transformations
%                from log-parameter to parameter


r = theta(1);
k = theta(2);
alpha = theta(3);
beta = theta(4);

r_prior = unifpdf(r,0,1);
k_prior = unifpdf(k,10,80);
alpha_prior = unifpdf(alpha,0,1);
beta_prior = unifpdf(beta,0,1);

  out = r_prior*k_prior * alpha_prior * beta_prior;
end
