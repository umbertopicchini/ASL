function summaries = mixture_sumstat(x,realdata)
% x is some kind of data, of type n x numsim, with n=number of cases and numsim the number of simulated datasets
% realdata is only used when computing summaries of observed data (realdata==1)
  

  summaries =zeros(4,1);
  
  if nargin==2 && realdata==1  % so here we retrieve observed summaries for actual data and exit
                               % essentially these are ground truth means
    % notice these are indeed sorted between mixture components to deal with label-switching issues
    % that is we want mean(v1)_c1 < mean(v1)_c2 and mean(v2)_c1 < mean(v2)_c2 
    % where v1 is variable 1, v2 is variable 2, c1 is component 1 in the Gaussian mixture and 
    % c2 is component 2 in the Gaussian mixture 
      summaries(1) = -5; % mean(v1)_c1
      summaries(2) = 10; % mean(v2)_c1
      summaries(3) = 30; % mean(v1)_c2
      summaries(4) = 20; % mean(v2)_c2
      return
  end
  
  options = statset('MaxIter',200);
  fit = fitgmdist(x,2,'Options',options);  % fit a 2-components gaussian mixture

  means = fit.mu;
 % means = means(:);
 % means = sort(means);
  means_component_1 = means(1,:);
  means_component_2 = means(2,:);
  
  % sort to deal with label-switching issues
   means_dimension_1 = sort([means_component_1(1),means_component_2(1)]);
   means_dimension_2 = sort([means_component_1(2),means_component_2(2)]);
  
  summaries(1,:) = means_dimension_1(1); % mean(v1)_c1
  summaries(2,:) = means_dimension_2(1); % mean(v2)_c1
  summaries(3,:) = means_dimension_1(2); % mean(v1)_c2
  summaries(4,:) = means_dimension_2(2); % mean(v2)_c2

 
end
