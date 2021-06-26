function sim_summ = recruitment_simsummaries(bigtheta,numsim)

% simunlate n x numsim "data points" iid from a Gaussian with mean 0.5 and standard deviation 0.05,
% truncated to the interval [0.01,1.2].

r = bigtheta(1);
k = bigtheta(2);
alpha = bigtheta(3);
beta = bigtheta(4);


n = 300; % but will only consider the last 250 generated values
n_transient = 50;
N = zeros(n,numsim);
N(1,:) = 10;
for ii=2:n
   flag = N(ii-1,:) <= k;
   index = find(flag);
   if sum(flag)>0
      N(ii,index) = poissrnd(N(ii-1,index)*(1+r)) + poissrnd(beta,1,sum(flag));
   end
   index2 = find(flag==0);
   if sum(flag==0)>0
     N(ii,index2) = binornd(N(ii-1,index2),alpha) + poissrnd(beta,1,sum(flag==0));
   end
end

% remove the first 50 observations (transient phase) as in Fasiolo et al.
% "An Extended Empirical Saddlepoint Approximation forIntractable Likelihoods"

N = N(n_transient+1:end,:);

nobs = size(N,1);
d = diff(N);
ratio = (N(2:nobs,:)+1)./(N(1:nobs-1,:)+1);

sim_summ = zeros(12,numsim);

sim_summ(1,:) = mean(N);
sim_summ(2,:) = var(N);
sim_summ(3,:) = skewness(N);
sim_summ(4,:) = kurtosis(N);
sim_summ(5,:) = mean(d);
sim_summ(6,:) = var(d);
sim_summ(7,:) = skewness(d);
sim_summ(8,:) = kurtosis(d);
sim_summ(9,:) = mean(ratio);
sim_summ(10,:) = var(ratio);
sim_summ(11,:) = skewness(ratio);
sim_summ(12,:) = kurtosis(ratio);

end

