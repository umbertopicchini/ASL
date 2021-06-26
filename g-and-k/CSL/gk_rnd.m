function y= gk_rnd(theta,z)

% z is a realization from z = randn(n,numsim), where n=number of observations
% numsim is number of simulated datasets


  logA = theta(1);
  logB = theta(2);
  logg = theta(3);
  logk = theta(4);

  A = exp(logA);
  B = exp(logB);
  g = exp(logg);
  k = exp(logk);

  y = A + B * (1 + 0.8 * (1-exp(-g*z))./(1+exp(-g*z))) .* (1 + z.^2).^k .* z;


end
