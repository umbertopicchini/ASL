
rng(100)  

problem = 'alphastable';
% ground-truth parameters
alpha = 1.01;
beta = 0.5;
gamma = 1;
delta = 0;
covariates = gamma;

% transform parameters as in Ong et al https://doi.org/10.1007/s11222-017-9773-3
alphatilde = log((alpha-0.5)/(2-alpha));
betatilde = log((beta+1)/(1-beta));
gammatilde = log(gamma);
deltatilde = delta;

nobs = 500;  % data sample size
thetatilde_true = [alphatilde, betatilde, gammatilde, deltatilde];  % ground-truth parameters


numsim= 1000;  
numattempts = 5;
mcwm=1;


% generate data
% generate from an alpha-stable as in Peters, G. W., Sisson, S. A., & Fan, Y. (2012). Likelihood-free Bayesian inference for α-stable models. Computational Statistics & Data Analysis, 56(11), 3743-3756.
% see also Weron, A., & Weron, R. (1995). Computer simulation of Lévy α-stable variables and processes. In Chaos—The interplay between stochastic and deterministic behaviour (pp. 379-392). Springer, Berlin, Heidelberg.
W = exprnd(1,nobs,1);
U = unifrnd(-pi/2,pi/2,nobs,1);

if alpha>=0.97 && alpha<= 1.03
    ybar = 2/pi * ((pi/2+beta*U).*tan(U) -beta * log( (W.*cos(U))./(pi/2+beta*U) ));
    simdata = gamma*ybar + (2/pi) * beta*gamma * log(gamma) + delta;
else
    S = (1+beta^2* (tan(pi*alpha/2))^2)^(1/(2*alpha));
    B = 1/ alpha * atan(beta*tan(pi*alpha/2));
    ybar = S * (sin((alpha)*(U+B))) ./ ((cos(U)).^(1/alpha)) .* ( cos(U-alpha*(U+B)) ./ W).^((1-alpha)/alpha) ;
    simdata = gamma*ybar + delta;
end


% Starting parameter values
alpha_start = 0.8;
beta_start = 0.95;
gamma_start = 2;
delta_start = 1;

alphatilde_start = log((alpha_start-0.5)/(2-alpha_start));
betatilde_start = log((beta_start+1)/(1-beta_start));
gammatilde_start = log(gamma_start);
deltatilde_start = delta_start;

thetatilde_start = [alphatilde_start, betatilde_start, gammatilde_start, deltatilde_start];


for attempt= 1:numattempts
    attempt
Haario_mcmciter = 5200;
Haario_proposal = diag([0.01, 0.01, 0.01, 0.01].^2);
Haario_length_CoVupdate = 30;
Haario_burnin = 200;
chains_withHaario = bslmcmc(problem,thetatilde_start,y,covariates,numsim,Haario_mcmciter,Haario_proposal,Haario_burnin,Haario_length_CoVupdate,mcwm);
% save back to the original scale
alpha_chain = (0.5 + 2*exp(chains_withHaario(:,1))) ./ (1+exp(chains_withHaario(:,1)));
beta_chain = (exp(chains_withHaario(:,2))-1)./(1+exp(chains_withHaario(:,2)));
gamma_chain = exp(chains_withHaario(:,3));
delta_chain = chains_withHaario(:,4);
MCMC = [alpha_chain,beta_chain,gamma_chain,delta_chain];
filename = sprintf('chains_attempt%d',attempt);
save(filename,'MCMC','-ascii')
end

