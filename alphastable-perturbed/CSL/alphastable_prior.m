function out = alphastable_prior(thetatilde)

% Returns the product of independent priors for parameters of a g-and-k distribution
% Input:  - theta, the vector of parameters to be estimated
% Output: - out, the product of the prior distributions set for each parameter.

% thetatilde are transformed parameters as in Ong et al https://doi.org/10.1007/s11222-017-9773-3

alphatilde = thetatilde(1);
betatilde  = thetatilde(2);
gammatilde = thetatilde(3);
deltatilde = thetatilde(4);


  alphatilde_prior      = normpdf(alphatilde);
  betatilde_prior      = normpdf(betatilde);
  gammatilde_prior      = normpdf(gammatilde);
  deltatilde_prior      = normpdf(deltatilde);

  out = alphatilde_prior*betatilde_prior*gammatilde_prior*deltatilde_prior;
end
