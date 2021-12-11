function out = mixture_prior(theta)

% Input:  - theta, the vector of parameters to be estimated
% Output: - out, the product of the prior distributions set for each parameter.

  theta = theta(:);
  
  theta_dimension_1 = sort([theta(1),theta(3)]);
  theta_dimension_2 = sort([theta(2),theta(4)]);
  
  m1 = theta_dimension_1(1);
  m2 = theta_dimension_2(1);
  m3 = theta_dimension_1(2);
  m4 = theta_dimension_2(2);



  m1_prior      = normpdf(m1,-5,2);
  m2_prior      = normpdf(m2, 10,2);
  m3_prior      = normpdf(m3,30,2);
  m4_prior      = normpdf(m4,20,2);


  out = m1_prior*m2_prior*m3_prior*m4_prior;
end
