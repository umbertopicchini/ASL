%rng(872223430)
rng(100)

r = 0.4;
k = 50;
alpha = 0.09;
beta = 0.05;

theta_true =[r,k,alpha,beta];
% simulate "observed" summaries
s_obs = recruitment_simsummaries(theta_true,1);


%theta_start = [  0.8     65    0.05   0.07];  % startset 1
theta_start = [  1     75    0.02   0.07];  % startset 2

numsim= 200;  % number of summaries simulated using SL
mcmciter= 2000; % total number of iterations
sd_randomwalk = [0.005, 0.5, 0.001, 0.001];  % std deviation for Gaussian random walk during burnin
burnin = 500;  
robust = 0; % if 1 uses a procedure robust against lack-of Gaussianity for the summaries. If 0 uses the standard BSL
if mod(burnin,2)>0
    error('BURNIN must be even.')
end
mcwm = 1;
lengthcovupdate = 50; 
randomwalk_iterstart = 800; % the first post-burnin iteration when random-walk metropolis kicks-in
verbose = 1;

chains = aslmcmc(theta_start,s_obs,numsim,mcmciter,sd_randomwalk,burnin,randomwalk_iterstart,verbose,robust,mcwm,lengthcovupdate);
save('chains','chains','-ascii');
