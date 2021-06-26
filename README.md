# ASL: Sequentially guided MCMC proposals for synthetic likelihoods and correlated synthetic likelihoods

This is supporting code for version V3 (28 June 2021) of the paper by Umberto Picchini, Umberto Simola and Jukka Corander "Sequentially guided MCMC proposals for synthetic likelihoods and correlated synthetic likelihoods", http://arxiv.org/abs/2004.04558 (previous versions were named "Adaptive MCMC for synthetic likelihoods and correlated synthetic likelihoods")

- the "g-and-k" folder pertainssection 6.1, and specifically:
    - subfolder ASL considers the novel adaptive MCMC for synthetic likelihoods
    - subfolder BSL-Haario considers standard BSL with the adaptive MCMC method of Haario et al.
    - subfolder CSL considers correlated synthetic likelihoods with the adaptive MCMC method of Haario et al.
    - subfolder ELFI is Python code meant to run with the ELFI engine https://elfi.readthedocs.io
- the "supernova" folder pertains section 6.2, and specifically
    - subfolder ASL considers the novel adaptive MCMC for synthetic likelihoods;
    - subfolder BOLFI_supernovae is Python code meant to run with the ELFI engine https://elfi.readthedocs.io
    - subfolder standard-BSL-Haario considers standard BSL with the adaptive MCMC method of Haario et al.
- the "recruitment" folder pertains the recruitment boom and bust example (section 6.3) and specifically the code runs the ASL method, except for
      the subfolder BSL-Haario_robust which implements semiparametric BSL with the adaptive MCMC method of Haario et al.

     
