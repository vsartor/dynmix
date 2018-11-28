'''
This module implements a simple level-based DMMM, with first-order
polynomial DLMs for each of the k clusters.

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
from . import dirichlet


def estimator(y, k, init_level, delta = 0.9, numit = 100):
    '''
    Simple Dynamic Membership Mixture Model. A level-based mixture
    model with a first-order polynomial DLM for each cluster. This
    variation of the function attempts to perform point estimates.

    Args:
        y: An array with each row being the time-series from one
            observational unit.
        k: Number of clusters.
        init_level: The initial level of each cluster.
        delta: The universal discount factor to be used for all the
            units.
        numit: Number of iterations for the algorithm to run.

    Returns:
        The evolution of the estimates for each parameter.
    '''

    n, T = y.shape

    #-- Initialize the parameters

    # DLM parameters
    phi = np.ones(k)
    phi_w = np.ones(k)
    theta = np.tile(init_level, (T, 1)).T

    # Dirichlet Process parameters
    eta = np.tile(npr.dirichlet(np.ones(k)), (n, T, 1))
    Z = np.tile(npr.multinomial(1, np.ones(k)/k), (n, T, 1))

    #-- Constants

    F = G = np.array([[1]])

    #-- Iterative updates of parameter estimates based on means

    for _ in range(numit):
        # Update membership dummy parameters for each unit
        for i in range(n):
            for t in range(T):
                probs = sps.norm.pdf(y[i,t], theta[:,t], 1. / np.sqrt(phi))
                params = eta[i,t] * probs
                Z[i,t] = np.zeros(k)
                Z[i,t,params.argmax()] = 1

        # Update Dirichlet states for each unit
        for i in range(n):
            c = dirichlet.forward_filter(Z[i], delta, np.ones(k) * 0.1)
            eta[i] = dirichlet.backwards_estimator(c, delta)

        # Update DLM states and parameters for each cluster
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = [y[Z[:,t,j] == 1,t] for t in range(T)]

            # Update states
            V = np.array([[1 / np.sqrt(phi[j])]])
            W = np.array([[1 / np.sqrt(phi_w[j])]])
            filters = dlm.multi_filter(YJ, F, G, V, W)
            theta[j] = dlm.smoother(G, *filters)[0][:,0]

            # Update parameters
            observation_ssq = 0
            for t in range(T):
                observation_ssq += np.sum((YJ[t] - theta[j,t])**2)
            phi[j] = n / observation_ssq
            phi_w[j] = np.mean((theta[j,:-1] - theta[j,1:])**2)

    return eta, theta, phi, phi_w


def sampler(y, k, delta, init_level, numit=5000, burnin=1000):
    '''
    Simple Dynamic Membership Mixture Model. A level-based mixture
    model with a first-order polynomial DLM for each cluster. This
    variation of the function performs MCMC sampling for the
    posterior distribution..

    Args:
        y: An array with each row being the time-series from one
            observational unit.
        k: Number of clusters.
        delta: The universal discount factor to be used for all the
            units.
        init_level: The initial level of each cluster.
        numit: Number of iterations for the algorithm to run.
        burnin: Number of initial iterations to be discarded.

    Returns:
        Samples for each parameter.
    '''

    n, T = y.shape

    #-- Allocate the chains

    phi = np.empty((numit, k))
    phi_w = np.empty((numit, k))
    theta = np.empty((numit, k, T))

    eta = np.empty((numit, n, T, k))
    Z = np.empty((numit, n, T, k))

    #-- Initialize the parameters

    phi[0] = np.ones(k)
    phi_w[0] = np.ones(k)
    theta[0] = np.tile(init_level, (T, 1)).T

    # Dirichlet Process parameters
    eta[0] = np.tile(npr.dirichlet(np.ones(k)), (n, T, 1))
    Z[0] = np.tile(npr.multinomial(1, np.ones(k)/k), (n, T, 1))

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
            for t in range(T):
                f_vals = sps.norm.pdf(y[i,t], theta[l-1,:,t], sd)
                weights = eta[l-1,i,t] * f_vals
                Z[l,i,t] = npr.multinomial(1, weights / weights.sum())

        # Sample Dirichlet states for each unit
        for i in range(n):
            c = dirichlet.forward_filter(Z[l,i], delta, c0)
            eta[l,i] = dirichlet.backwards_sampler(c, delta)

        # Sample DLM states and parameters for each cluster
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = [y[Z[l,:,t,j] == 1,t] for t in range(T)]

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
