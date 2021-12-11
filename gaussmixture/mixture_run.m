
rng(1234) 

% Features of the 2 components, bidimensional gaussian micture
mu = [-5 10; 30 20];
sigma1 = [4^2 0; 0 4^2];
sigma2 = [4^2 12; 12 4^2];
sigma(:,:,1) = sigma1;
sigma(:,:,2) = sigma2;
prop = [1/2, 1/2];

gm = gmdistribution(mu,sigma,prop);

% plot the mixture
gmPDF = @(x,y) arrayfun(@(x0,y0) pdf(gm,[x0 y0]),x,y);
figure
fsurf(gmPDF,[-20 40])


figure
fcontour(gmPDF,[-20 40]);

nobs = 5000;  % data sample size
% generate data
data = random(gm,nobs);
scatter(data(:,1),data(:,2),10,'.') % Scatter plot with points of size 10
hold on
fcontour(gmPDF,[-30 50]);


numsim= 10;  
mcmciter= 100;
sd_randomwalk = 0.2*ones(1,4);
burnin = 50;  % must be even
Haario_length_CoVupdate = 10;
randomwalk_iterstart = 80; % the first post-burnin iteration when random-walk metropolis kicks-in
verbose = 1;
mcwm=1;  % if mcwm==1 we use BSL with MCWM sampling during the burnin iterations (NOT afterwards). If mcwm==0 we use standard BSL MCMC during burnin
numattempts = 100;

if mod(burnin,2)>0
    error('BURNIN must be even.')
end
if randomwalk_iterstart <= burnin
    error('RANDOMWALK_ITERSTART should be larger than burnin')
end
if burnin < Haario_length_CoVupdate
    error('BURNIN must be larger than Haario_length_CoVupdate')
end


for attempt= 1:numattempts
    attempt
thetastart = unifrnd(-30,50,1,4); % generates 4 random numbers uniformly in (-30,50)
[chains,proposal_cov] = aslmcmc(thetastart,data,numsim,mcmciter,sd_randomwalk,Haario_length_CoVupdate,burnin,randomwalk_iterstart,verbose,mcwm);

filename = sprintf('chains_attempt%d',attempt);
save(filename,'chains','-ascii')
end

