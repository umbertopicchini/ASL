# ASL: Sequentially guided MCMC proposals for synthetic likelihoods and correlated synthetic likelihoods

This is supporting code for version V4 of the paper by Umberto Picchini, Umberto Simola and Jukka Corander "Sequentially guided MCMC proposals for synthetic likelihoods and correlated synthetic likelihoods", http://arxiv.org/abs/2004.04558 (previous versions were named "Adaptive MCMC for synthetic likelihoods and correlated synthetic likelihoods")

- "alphastable-perturbed"
    - subfolder BSL considers standard BSL with the adaptive MCMC method of Haario et al.
    - subfolder CSL considers correlated synthetic likelihoods with the adaptive MCMC method of Haario et al.
- "g-and-k":
    - subfolder ASL considers the sequentially guided adaptive MCMC for synthetic likelihoods
    - subfolder BSL-Haario considers standard BSL with the adaptive MCMC method of Haario et al.
    - subfolder CSL considers correlated synthetic likelihoods with the adaptive MCMC method of Haario et al.
    - subfolder ELFI is Python code meant to run with the ELFI engine https://elfi.readthedocs.io
- the "supernova" folder:
    - subfolder ASL considers the sequentially guided  adaptive MCMC for synthetic likelihoods;
    - subfolder BOLFI_supernovae is Python code meant to run with the ELFI engine https://elfi.readthedocs.io
    - subfolder standard-BSL-Haario considers standard BSL with the adaptive MCMC method of Haario et al.
- the "recruitment" folder:
    - subfolder ASL considers the sequentially guided  adaptive MCMC for synthetic likelihoods;
    - the subfolder BSL-Haario_robust implements semiparametric BSL with the adaptive MCMC method of Haario et al.
- the "gaussmixture" folder:
    it shows that ASL rapidly discovers the multiple modes of a Gaussian mixture

     
