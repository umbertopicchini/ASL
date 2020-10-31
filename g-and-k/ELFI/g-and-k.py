# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:16:47 2018

@author: picchini
"""

import numpy as np
import scipy.stats as ss
from scipy.io import loadmat
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




def GNK(A, B, g, k, n_obs, c=0.8, batch_size=1, random_state=None):
    """Sample the univariate g-and-k distribution.

    References
    ----------
    [1] Drovandi, C. C., & Pettitt, A. N. (2011).
    Likelihood-free Bayesian estimation of multivariate quantile distributions.
    Computational Statistics & Data Analysis, 55(9), 2541-2556.
    [2] Allingham, D., King, R. A. R., & Mengersen, K. L. (2009).
    Bayesian estimation of quantile distributions.
    Statistics and Computing, 19(2), 189-201.

    The quantile function of g-and-k distribution is defined as follows:

    Q_{gnk} = A + B * (1 + c * (1 - exp(-g * z(p)) / 1 + exp(-g * z(p))))
            * (1 + z(p)^2)^k * z(p), where

    z(p) is the p-th standard normal quantile.

    To sample from the g-and-k distribution, draw z(p) ~ N(0, 1) and evaluate Q_{gnk}.

    Parameters
    ----------
    A : float or array_like
        Location parameter.
    B : float or array_like
        Scale parameter.
    g : float or array_like
        Skewness parameter.
    k : float or array_like
        Kurtosis parameter.
    c : float, optional
        Overall asymmetry parameter, by default fixed to 0.8 as in Allingham et al. (2009).
    n_obs : int, optional
    batch_size : int, optional
    random_state : np.random.RandomState, optional

    Returns
    -------
    array_like
        Yielded points (the array's shape corresponds to (batch_size, n_points, n_dims).

    """
    # Transforming the arrays' shape to be compatible with batching.
    A = np.asanyarray(A).reshape((-1, 1))
    B = np.asanyarray(B).reshape((-1, 1))
    g = np.asanyarray(g).reshape((-1, 1))
    k = np.asanyarray(k).reshape((-1, 1))

    # Obtaining z(p) ~ N(0, 1).
    z = ss.norm.rvs(size=(batch_size, n_obs), random_state=random_state)

    # Evaluating the quantile function Q_{gnk}.
    y = A + B * (1 + c * ((1 - np.exp(-g * z)) / (1 + np.exp(-g * z)))) * (1 + z**2)**k * z

    # Dedicating a dummy axis for the dimensionality of the points.
    y = y[:, :, np.newaxis]
    return y

# true parameters
A_true = 3
B_true = 1
g_true = 2
k_true = 0.5

#n_obs = 1000
# generate data
#y_obs = GNK(A_true, B_true, g_true , k_true, n_obs)

# load data.
# gk_data.mat has n_obs=1000 and was generated in Matlab using the ground-truth parameters as given above and by setting a random seed with rng(1234)

matdata=loadmat('gk_data.mat')
y_obs=matdata['y']
y_obs = y_obs[:,:,np.newaxis]
n_obs = y_obs.size

# assign priors
A = elfi.Prior('uniform', -10, 10)
B  = elfi.Prior('uniform', 0, 10)
g  = elfi.Prior('uniform', 0, 10)
k  = elfi.Prior('uniform', 0, 10)

Y = elfi.Simulator(GNK, A,B,g,k, n_obs, observed=y_obs)

# summary statistics as in Drovandi and Petitt
def ss_robust(y):
    """Obtain the robust summary statistic described in Drovandi and Pettitt (2011).

    The statistic reaches the optimal performance upon a high number of
    observations.

    Parameters
    ----------
    y : array_like
        Yielded points.

    Returns
    -------
    array_like of the shape (batch_size, dim_ss=4, dim_ss_point)

    """
    ss_A = _get_ss_A(y)
    ss_B = _get_ss_B(y)
    ss_g = _get_ss_g(y)
    ss_k = _get_ss_k(y)

    # Combining the summary statistics.
    ss_robust = np.hstack((ss_A, ss_B, ss_g, ss_k))
    ss_robust = ss_robust[:, :, np.newaxis]
    return ss_robust

def _get_ss_A(y,weight=1):
    L2 = np.percentile(y, 50, axis=1)
    ss_A = L2 * weight
    return ss_A


def _get_ss_B(y,weight=1):
    L1, L3 = np.percentile(y, [25, 75], axis=1)

    # Avoiding the zero value (ss_B is used for division).
    ss_B = weight*(L3 - L1).ravel()
    idxs_zero = np.where(ss_B == 0)[0]
    ss_B[idxs_zero] += np.finfo(float).eps

    # Transforming the summary statistics back into the compatible shape.
    n_dim = y.shape[-1]
    n_batches = y.shape[0]
    ss_B = ss_B.reshape(n_batches, n_dim)
    return ss_B


def _get_ss_g(y,weight=1):
    L1, L2, L3 = np.percentile(y, [25, 50, 75], axis=1)
    ss_B = _get_ss_B(y)
    ss_g = np.divide(L3 + L1 - 2 * L2, ss_B)
    ss_g = ss_g * weight
    return ss_g


def _get_ss_k(y,weight=1):
    E1, E3, E5, E7 = np.percentile(y, [12.5, 37.5, 62.5, 87.5], axis=1)
    ss_B = _get_ss_B(y)
    ss_k = np.divide(E7 - E5 + E3 - E1, ss_B)
    ss_k = ss_k * weight
    return ss_k

SA = elfi.Summary(_get_ss_A, Y)
SB = elfi.Summary(_get_ss_B, Y)
Sg = elfi.Summary(_get_ss_g, Y)
Sk = elfi.Summary(_get_ss_k, Y)


# pilot to collect summaries and obtaine their relative weights
#SA_pilot= elfi.Summary.generate(SA, batch_size=1000)
#SB_pilot= elfi.Summary.generate(SB, batch_size=1000)
#Sg_pilot= elfi.Summary.generate(Sg, batch_size=1000)
#Sk_pilot= elfi.Summary.generate(Sk, batch_size=1000)

#from numpy import median, absolute

#def mad(data, axis=None):
#    return median(absolute(data - median(data, axis)), axis)

#SA_mad = mad(SA_pilot)
#SB_mad = mad(SB_pilot)
#Sg_mad = mad(Sg_pilot)
#Sk_mad = mad(Sk_pilot)

#SA_weighted = elfi.Summary(_get_ss_A, Y, 1/SA_mad)
#SB_weighted = elfi.Summary(_get_ss_B, Y, 1/SB_mad)
#Sg_weighted = elfi.Summary(_get_ss_g, Y, 1/Sg_mad)
#Sk_weighted = elfi.Summary(_get_ss_k, Y, 1/Sk_mad)


# Finish the model with the final node that calculates the squared distance (S1_sim-S1_obs)**2 + (S2_sim-S2_obs)**2
#d = elfi.Distance('euclidean', SA_weighted, SB_weighted, Sg_weighted, Sk_weighted)

#rej = elfi.Rejection(d, batch_size=10000, seed=seed)

#N=1000
#%time result = rej.sample(N, quantile=0.01)

#result.summary()

#smc = elfi.SMC(d, batch_size=1000, seed=seed)

#N = 500
#schedule = [0.7, 0.2, 0.05]
#%time result_smc = smc.sample(N, schedule)

#%time result2 = rej.sample(N, threshold=3)

d = elfi.Distance('euclidean', SA, SB, Sg, Sk)

log_d = elfi.Operation(np.log, d)

bolfi = elfi.BOLFI(log_d, batch_size=1, initial_evidence=20, update_interval=10, acq_noise_var=[0.1, 0.1, 0.1, 0.1], bounds={'A':(-10, 10), 'B':(0, 10), 'g':(0, 10), 'k':(0, 10)}, seed=seed)

%time post = bolfi.fit(n_evidence=500)

bolfi.plot_state();

bolfi.target_model

bolfi.plot_discrepancy();

bolfi.extract_result().x_min



#%time result_BOLFI = bolfi.sample(1000, n_chains=2, target_prob=0.8, info_freq=1000)
#result_BOLFI
#result_BOLFI.plot_traces();
