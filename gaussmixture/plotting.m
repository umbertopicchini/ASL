
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
fsurf(gmPDF,[-30 50])


figure
fcontour(gmPDF,[-30 50]);
hold on

for ii=1:100
    filename = sprintf('chains_attempt%d',ii);
    chains = load(filename);
    
    plot(chains(1,1),chains(1,2),'ko')
    plot(chains(1,3),chains(1,4),'ko')
    plot(chains(49,1),chains(49,2),'mo')
    plot(chains(49,3),chains(49,4),'mo')
    plot(chains(50,1),chains(50,2),'g*')
    plot(chains(50,3),chains(50,4),'g*')
end

