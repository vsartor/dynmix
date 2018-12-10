'''
This module implements a simple level-based time-series mixture model using
first-order polynomial DLMs for each of the k clusters. The diference between
this and the SDMMM is that cluster membership is static.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import numpy.random as npr
import scipy.stats as sps

from . import dlm


def sampler(y, k, init_level, numit=4500, burnin=500):
    '''
    TODO: Write documentation.
    '''

    n, T = y.shape

    #-- Allocate the chains

    phi = np.empty((numit, k))
    phi_w = np.empty((numit, k))
    theta = np.empty((numit, k, T))

    eta = np.empty((numit, n, k))
    Z = np.empty((numit, n, k))

    #-- Initialize the parameters

    phi[0] = np.ones(k)
    phi_w[0] = np.ones(k)
    theta[0] = np.tile(init_level, (T, 1)).T

    eta[0] = np.tile(npr.dirichlet(np.ones(k)), (n, 1))
    Z[0] = np.tile(npr.multinomial(1, np.ones(k)/k), (n, 1))

    # Make sure each cluster starts with an observation
    for i in range(k):
        Z[0,i] = np.zeros(k)
        Z[0,i,i] = 1

    #-- Constants

    F = G = np.array([[1]])
    c0 = np.ones(k) * 0.1

    #-- MCMC iterations

    for l in range(1, numit):
        if l % 20 == 0:
            print(f'Gibbs sampler at iteration {l} out of {numit}')

        sd = 1. / np.sqrt(phi[l-1])
        sd_w = 1. / np.sqrt(phi_w[l-1])

        # Sample membership dummy parameters for each unit
        for i in range(n):
            #TODO: Check this hehehehe
            weights = 0
            for t in range(T):
                f_vals = sps.norm.pdf(y[i,t], theta[l-1,:,t], sd)
                weights += eta[l-1,i] * f_vals
            Z[l,i] = npr.multinomial(1, weights / weights.sum())

        # Sample membership coefficients for each unit
        for i in range(n):
            eta[l,i] = npr.dirichlet(c0 + Z[l,i])

        # Sample DLM states and parameters for each cluster
        #TODO: Stop using multi_filter because it's unecessary here.
        #      Only using it right now for convenience of copy & paste
        #      from the SDMMM file.
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = [y[Z[l,:,j] == 1,t] for t in range(T)]

            # Sample states
            V = np.array([[sd[j]]])
            W = np.array([[sd_w[j]]])
            filters = dlm.multi_filter(YJ, F, G, V, W)
            s, S = dlm.smoother(G, *filters)
            theta[l,j] = npr.normal(s[:,0], S[:,0,0])

            # Sample observational precision
            num_obs = 0.0001
            observation_ssq = 0.0001
            for t in range(T):
                num_obs += 1
                observation_ssq += np.sum((YJ[t] - theta[l,j,t])**2)
            phi[l,j] = npr.gamma(num_obs / 2., 2. / observation_ssq)

            # Sample evolutional precision
            num_state = T - 1 + 0.0001
            state_ssq = np.sum((theta[l,j,:-1] - theta[l,j,1:])**2) + 0.0001
            phi_w[l,j] = npr.gamma(num_state / 2., 2. / state_ssq)

    return (eta[burnin:], Z[burnin:], theta[burnin:],
            phi[burnin:], phi_w[burnin:])
