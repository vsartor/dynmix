'''
This module implements a simple level-based time-series mixture model for
clustering using first-order polynomial DLMs for each of the k clusters.

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


def sampler(y, k, init_level, numit=4500, burnin=500, label_at=0):
    '''
    Obtains samples from the posterior of a simple mixture model
    with a first-order polynomial DLM determining behavior for each
    cluster.
    '''

    # Constants
    n, T = y.shape
    F = G = np.array([[1]])
    c0 = np.ones(k) * 0.1

    # Allocate the DLM parameter chains
    phi = np.empty((numit, k))
    phi_w = np.empty((numit, k))
    theta = np.empty((numit, k, T))

    # Allocate the DP parameter chains
    eta = np.empty((numit, n, k))
    Z = np.empty((numit, n, k))

    # Initialize parameters
    phi[0] = np.ones(k) * 0.01
    phi_w[0] = np.ones(k) * 0.01
    theta = np.empty((numit, k, T))
    for j in range(k):
        theta[0, j] = y[init_level[j]]
    eta[0] = np.tile(npr.dirichlet(np.ones(k)), (n, 1))
    Z[0] = np.tile(npr.multinomial(1, np.ones(k)/k), (n, 1))

    # Make sure the label_at ordering is set from the beginning
    order = np.argsort(theta[0, :, label_at])
    theta[0] = theta[0, order]

    # Note that "order" is the order at which the indexes should be
    # taken to sort it. For example:
    # a = [9 3 5]
    # order = [1 2 0]
    # That means that if we argsort the order we get
    # oo = [2 0 1]
    # Which gives us at the i-th entry the position that the
    # i-th entry of 'a' now belogs in a sorted manner.
    # Id est, because at 0-th we have 2, then the 0-th member
    # of a went to the 2-nd position!
    # We use this to set the init_level members to their appropriate
    # clusters.

    # Guarantee that each init_level member is assigned to the
    # respective cluster
    switch = np.argsort(order)
    for j, i in enumerate(init_level):
        Z[0, i] = np.zeros(k)
        Z[0, i, switch[j]] = 1

    # Gibbs sampler
    for l in range(1, numit):
        if l % 500 == 0:
            print(f'Gibbs sampler at iteration {l} out of {numit}')

        sd = 1. / np.sqrt(phi[l-1])
        sd_w = 1. / np.sqrt(phi_w[l-1])

        # Sample membership dummy parameters for each unit
        for i in range(n):
            logweights = np.log(eta[l-1, i])
            for t in range(T):
                logpdf_vals = sps.norm.logpdf(y[i, t], theta[l-1, :, t], sd)
                logweights += logpdf_vals
            weights = np.exp(logweights)

            # If sum is zero, approximate using the highest
            if weights.sum() == 0:
                print('Fixing null weights!')
                weights = np.zeros(k)
                weights[np.argmax(logweights)] = 1

            Z[l, i] = npr.multinomial(1, weights / weights.sum())

        if np.any(Z[l].sum(axis=0) == 0):
            raise ValueError("Empty clusters.")

        # Sample membership coefficients for each unit
        for i in range(n):
            eta[l, i] = npr.dirichlet(c0 + Z[l, i])

        # Sample DLM states and parameters for each cluster
        # TODO: Stop using multi_filter because it's unecessary here.
        #       Only using it right now for convenience of copy & paste
        #       from the unilevel file.
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = [y[Z[l, :, j] == 1, t] for t in range(T)]

            # Sample states
            V = np.array([[sd[j] ** 2]])
            W = np.array([[sd_w[j] ** 2]])
            filters = dlm.multi_filter(YJ, F, G, V, W)
            s, S = dlm.smoother(G, *filters)
            if np.any(np.sqrt(S[:, 0, 0]) <= 0):
                print('Negative variance out of smoother! Error incoming!')
                print(f'Values for S are: {S[:, 0, 0]}')
                print(f'V {V} W {W}')
            theta[l, j] = npr.normal(s[:, 0], np.sqrt(S[:, 0, 0]))

        # Deal with label switching
        order = np.argsort(theta[l, :, label_at])
        theta[l] = theta[l, order]

        # Continue sampling DLM states and parameters for each cluster
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = [y[Z[l, :, j] == 1, t] for t in range(T)]

            # Sample observational precision
            num_obs = 0.002
            observation_ssq = 0.002
            for t in range(T):
                num_obs += 1
                observation_ssq += np.sum((YJ[t] - theta[l, j, t])**2)
            phi[l, j] = npr.gamma(num_obs / 2., 2. / observation_ssq)

            # Sample evolutional precision
            num_state = T - 1 + 0.002
            state_ssq = \
                np.sum((theta[l, j, :-1] - theta[l, j, 1:])**2) + 0.002
            phi_w[l, j] = npr.gamma(num_state / 2., 2. / state_ssq)

    return (eta[burnin:], Z[burnin:], theta[burnin:],
            phi[burnin:], phi_w[burnin:])
