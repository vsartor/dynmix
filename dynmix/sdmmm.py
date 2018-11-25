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

from .dlm import dlm_multi_filter, dlm_smoother
from .dirichlet import dirichlet_forward_filter, dirichlet_backwards_estimator


def sdmmm_likelihood(y, Z, eta, delta, theta, phi, phi_w):
    '''
    Likelihood function for the SDMMM.

    Args:
        y: An array with each row being the time-series from one
            observational unit.
        Z: Array with each row being a matrix with the membership
            dummy variable for one observational unit.
        eta: Array with each row being a matrix with the states from
            the Dirichlet process for one observational unit.
        delta: The universal discount factor to be used for all the
            units.
        theta: Array with each row being the state-series for one
            cluster.
        phi: Array with the observational precision for each cluster.
        phi_w: Array with the evolutional precision for each cluster.
    '''

    #TODO: Implement

    return 0


def sdmmm_estimator(y, k, delta = 0.9, numit = 100):
    '''
    Simple Dynamic Membership Mixture Model. A level-based mixture
    model with a first-order polynomial DLM for each cluster. This
    variation of the function attempts to perform point estimates.

    Args:
        y: An array with each row being the time-series from one
            observational unit.
        k: Number of clusters.
        delta: The universal discount factor to be used for all the
            units.
        numit: Number of iterations for the algorithm to run.

    Returns:
        The evolution of the estimates for each parameter.
    '''

    n, T = y.shape

    #-- Initialize the parameters
    #TODO: Allow user initializations

    # DLM parameters
    phi = np.ones(k)
    phi_w = np.ones(k)
    theta = np.tile(npr.normal(0, 20, k), (T, 1)).T

    # Dirichlet Process parameters
    eta = np.tile(npr.dirichlet(np.ones(k)), (n, T, 1))
    Z = np.tile(npr.multinomial(1, np.ones(k)/k), (n, T, 1))

    # To make sure at least one observation is in each cluster, manually set
    # each of the first k units into each one of the k clusters
    for i in range(k):
        Z[i] = np.zeros((T, k))
        Z[i,:,i] = np.ones(T)

    # Likelihood
    U = np.empty(numit)

    #-- Constants

    F = G = np.array([[1]])

    #-- Iterative updates of parameter estimates based on means

    for l in range(numit):
        print(f'sdmmm_estimator: iteration {i} out of {numit}')

        # Update membership dummy parameters for each unit
        for i in range(n):
            for t in range(T):
                probs = sps.norm.pdf(y[i,t], theta[:,t], 1. / np.sqrt(phi[k]))
                params = eta * probs
                Z[i,t] = np.zeros(k)
                Z[i,params.argmax()] = 1

        # Update Dirichlet states for each unit
        for i in range(n):
            c = dirichlet_forward_filter(Z, delta, np.ones(k) * 0.1)
            eta = dirichlet_backwards_estimator(c, delta)

        # Update DLM states and parameters for each cluster
        for j in range(k):
            # Create observation list for multi_dlm
            YJ = []
            for t in range(T):
                mask = Z[:,t,j] == 1
                YJ.append(y[mask,t])

            # Update states
            V = np.array([[1 / np.sqrt(phi)]])
            W = np.array([[1 / np.sqrt(phi_w)]])
            filters = dlm_multi_filter(YJ, F, G, V, W)
            theta[j], _ = dlm_smoother(G, *filters)

            # Update parameters
            observation_ssq = 0
            for t in range(T):
                observation_ssq += np.sum((YJ[t] - theta[j,t])**2)
            phi = n / observation_ssq
            phi_w = np.mean((theta[k,:-1] - theta[k,1:])**2)

        # Update likelihood
        # U[l] = sdmmm_likelihood(y, Z, eta, delta, theta, phi, phi_w)
        U[l] = 0.

    return eta, theta, phi, phi_w, U
