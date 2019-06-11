'''
This module implements an univariate Dynamic Clustering Model with
first-order polynomial DLMs determining cluster behavior.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import numpy.random as rng
import scipy.stats as sps

from . import dlm
from . import common
from . import dirichlet


def estimator(Y, F_list, G_list, delta, numit=10, mnumit=100, numeps=1e-6, M=200):
    '''
    Uses Expectation-Maximization to estimate dynamic clusterization of n
    m-variate time-series, all observed throughout the same T time instants.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        delta: Vector with discount factor for each observation.
        numit: Number of iterations for the algorithm to run.
        mnumit: Maximum number of iterations for the M-step algorithm to run.
        numeps: Numerical precision for the M-step algorithm.
        M: Number of Monte-Carlo simulations of dummy variables.

    Returns:
        eta: A list with the eta for each time-series.
        theta: A list with the theta for each cluster.
        phi: A list with the phi for each cluster.
        W: A list with the W for each cluster.
    '''

    #-- Preamble 

    k, _, _, n, T, _ = common.get_dimensions(Y, F_list, G_list)
    _, theta, phi, eta = common.initialize(Y, F_list, G_list, dynamic=True)
    
    c0 = np.ones(k)
    mc_estimates = np.empty((T, k))

    #-- Algorithm

    for it in range(numit):
        print(f'\nIteration {it} out of {numit}', end = '')
        
        # Step 0: Expectation step

        weights = common.compute_weights_dyn(Y, F_list, G_list, theta, phi, eta)
        
        # Step 1: Maximize the weights for each observation

        for i in range(n):
            print('.', end='')

            # Simulate M observations and obtain the mean of all M estimates
            mc_estimates[:,:] = 0
            for _ in range(M):
                mc_Y = np.array([rng.multinomial(1, x) for x in weights[:,i,:]])
                c = dirichlet.forward_filter(mc_Y, delta[i], c0)
                mc_estimates += dirichlet.backwards_estimator(c, delta[i]) / M
            eta[:,i,:] = mc_estimates

        # Step 2: Maximize the cluster parameters

        for j in range(k):
            theta[j], V, _, _ = dlm.dynamic_weighted_mle(Y, F_list[j], G_list[j], eta[:,:,j],
                                                         maxit=mnumit, numeps=numeps)
            phi[j,:] = 1 / np.diag(V)

    return eta, theta, phi


def sampler(y, k, delta, init_level, numit=4500, burnin=500):
    '''
    Obtain samples from the posterior of a simple univariate
    first-order polynomial DLM dynamic clustering.

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

    # Constants
    n, T = y.shape
    F = G = np.array([[1]])
    c0 = np.ones(k) * 0.1

    # Allocate cluster parameter chains
    phi = np.empty((numit, k))
    phi_w = np.empty((numit, k))
    theta = np.empty((numit, k, T))

    # Allocate Dirichlet Process parameter chains
    eta = np.empty((numit, n, T, k))
    Z = np.empty((numit, n, T, k))

    # Initialize using the point-estimates
    eta[0], Z[0], theta[0], phi[0], phi_w[0] = \
        estimator(y, k, delta, init_level)

    # Gibbs iterations
    for l in range(1, numit):
        if l % 500 == 0:
            print(f'Gibbs sampler at iteration {l} out of {numit}')

        sd = 1. / np.sqrt(phi[l-1])
        sd_w = 1. / np.sqrt(phi_w[l-1])

        # Sample membership dummy parameters for each unit
        for i in range(n):
            for t in range(T):
                # Does not need constant because it'll be scaled
                # f_vals = np.exp(-(y[i, t] - theta[l - 1, :, t])**2 / (2*sd**2))
                f_vals = sps.norm.pdf(y[i, t], theta[l-1, :, t], sd)
                weights = eta[l-1, i, t] * f_vals
                Z[l, i, t] = rng.multinomial(1, weights / weights.sum())

        # Sample Dirichlet states for each unit
        for i in range(n):
            c = dirichlet.forward_filter(Z[l, i], delta, c0)
            eta[l, i] = dirichlet.backwards_sampler(c, delta)

        # Sample DLM states and parameters for each cluster
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = [y[Z[l, :, t, j] == 1, t] for t in range(T)]

            # Sample states
            V = np.array([[sd[j] ** 2]])
            W = np.array([[sd_w[j] ** 2]])
            filters = dlm.multi_filter(YJ, F, G, V, W)
            s, S = dlm.smoother(G, *filters)
            theta[l, j] = rng.normal(s[:, 0], np.sqrt(S[:, 0, 0]))

            # Sample observational precision
            num_obs = 0.0001
            observation_ssq = 0.0001
            for t in range(T):
                num_obs += 1
                observation_ssq += np.sum((YJ[t] - theta[l, j, t])**2)
            phi[l, j] = rng.gamma(num_obs / 2., 2. / observation_ssq)

            # Sample evolutional precision
            num_state = T - 1 + 0.0001
            state_ssq = \
                np.sum((theta[l, j, :-1] - theta[l, j, 1:])**2) + 0.0001
            phi_w[l, j] = rng.gamma(num_state / 2., 2. / state_ssq)

    # Return chains dropping burnin phase
    return (eta[burnin:], Z[burnin:], theta[burnin:],
            phi[burnin:], phi_w[burnin:])
