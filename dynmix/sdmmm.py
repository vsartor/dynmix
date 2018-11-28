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


def logpdf(y, Z, eta, delta, theta, phi, phi_w):
    '''
    Log posterior function for the SDMMM.

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

    n, T = y.shape
    k = phi.size

    sd = 1 / np.sqrt(phi)
    sd_w = 1 / np.sqrt(phi_w)

    # 1. Likelihood: p(y|Z,theta,phi)

    ll = 0
    for i in range(n):
        for t in range(T):
            cluster = Z[i,t] == 1
            ll += sps.norm.logpdf(y[i,t], theta[cluster,t], sd[cluster])

    # 2. Dynamic Linear Models: p(theta|phi_w)

    ldlm = 0
    for j in range(k):
        ldlm += np.sum(sps.norm.logpdf(theta[j,1:], theta[j,:-1], sd_w[j]))

    # 3. Dummy Variables: p(Z|eta)

    ldummy = 0
    for i in range(n):
        for t in range(T):
            ldummy += sps.multinomial.logpmf(Z[i,t], 1, eta[i,t])

    # 4. Dirichlet Process: p(eta|delta)

    ldir = 0
    for i in range(n):
        for t in range(1,n):
            # TODO: This is not actually correct (or is it?)
            ldir += sps.dirichlet.logpdf(eta[i,t], delta * eta[i,t-1])

    return ll + ldlm + ldummy + ldir


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

    # Likelihood
    U = np.empty(numit)

    #-- Constants

    F = G = np.array([[1]])

    #-- Iterative updates of parameter estimates based on means

    for l in range(numit):
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

        # Update likelihood
        U[l] = logpdf(y, Z, eta, delta, theta, phi, phi_w)

    return eta, theta, phi, phi_w, U
