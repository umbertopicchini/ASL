# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 15:16:41 2018

@author: picchini
"""

import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
%precision 2

import logging
logging.basicConfig(level=logging.INFO) # sometimes this is required to enable logging inside Jupyter

# Set an arbitrary global seed to keep the randomly generated quantities the same
seed = 1
np.random.seed(seed)

import elfi

from elfi.examples import ma2
model = ma2.get_model(seed_obs=seed)
elfi.draw(model)

log_d = elfi.Operation(np.log, model['d'])

bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10, bounds={'t1':(-2, 2), 't2':(-1, 1)}, acq_noise_var=[0.1, 0.1], seed=seed)



%time post = bolfi.fit(n_evidence=300)

bolfi.target_model



bolfi.plot_state();

bolfi.plot_discrepancy();

post2 = bolfi.extract_posterior(-1.)


post.plot(logpdf=True)

%time result_BOLFI = bolfi.sample(1000, info_freq=1000)

result_BOLFI


result_BOLFI.plot_traces();

result_BOLFI.plot_marginals();






