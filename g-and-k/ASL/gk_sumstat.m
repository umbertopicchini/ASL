function summaries = gk_sumstat(x)
% x is some kind of data, of type n x numsim, with n=number of cases and numsim the number of simulated datasets

  octile = prctile(x,100*[0.125,0.25,0.375,0.50,0.625,0.75,0.875]);
  
  if size(x,2)==1  % x has only 1 column
      octile = octile';
  end
  
  s_A = octile(4,:);
  s_B = octile(6,:) - octile(2,:);
  s_g = (octile(6,:) + octile(2,:) - 2*s_A) ./s_B;
  s_k = (octile(7,:) - octile(5,:) + octile(3,:) - octile(1,:))./s_B;
  
  

  summaries =zeros(4, size(x,2));

  
  summaries(1,:) = log(s_A);
  summaries(2,:) = log(s_B);
  summaries(3,:) = log(s_g);
  summaries(4,:) = log(s_k);

 
end
