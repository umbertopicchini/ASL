function summaries = alphastable_sumstat(x,covariates)
% x is some kind of data, of type n x numsim, with n=number of cases and numsim the number of simulated datasets

q_95pct = prctile(x,95);
q_5pct = prctile(x,5);
q_25pct = prctile(x,25);
q_50pct = prctile(x,50);
q_75pct = prctile(x,75);

gamma_true = covariates;

summary_alpha = (q_95pct - q_5pct)./(q_75pct-q_25pct);
summary_beta = (q_95pct + q_5pct -2*q_50pct) ./ (q_95pct - q_5pct);
summary_gamma = (q_75pct-q_25pct) / gamma_true;
summary_delta = mean(x);

summaries = [summary_alpha;summary_beta;summary_gamma;summary_delta];


end
