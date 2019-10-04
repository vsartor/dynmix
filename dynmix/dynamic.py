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
from . import independent

from tqdm import tqdm
from typing import List


def estimator(Y: np.ndarray, F_list: List[np.ndarray], G_list: List[np.ndarray], model_delta: bool = False,
              numit: int = 10, mnumit:int = 100, numeps: float = 1e-6, M: int = 200):
    '''
    Uses Expectation-Maximization to estimate dynamic clusterization of n
    m-variate time-series, all observed throughout the same T time instants.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        model_delta: A boolean indicating if delta is being modelled.
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

    print('Initializing...', end=' ')
    s_eta, _, _, _ = independent.estimator(Y, F_list, G_list)
    delta = np.clip(np.max(np.mean(s_eta, axis=0), axis=1), 0.05, 0.95)
    print('Finished initialization.')

    c0 = np.ones(k)
    mc_estimates = np.empty((T, k))
    if model_delta:
        mc_deltas = np.empty(M)

    #-- Algorithm

    for it in range(numit):
        print(f'\nIteration {it+1} out of {numit}', end='')

        # Step 0: Expectation step

        weights = common.compute_weights_dyn(Y, F_list, G_list, theta, phi, eta)

        # Step 1: Maximize the weights for each observation

        for i in range(n):
            print('.', end='')

            # Simulate M observations and obtain the mean of all M estimates
            mc_estimates[:,:] = 0

            for l in range(M):
                if i > 20:
                    i = i + 1 - 1

                # Generate multinomial observations
                mc_Y = np.array([rng.multinomial(1, x) for x in weights[:,i,:]])
                if model_delta:
                    # If modelling delta, find maximum for delta and use it for c's computation
                    mc_Z = np.argmax(mc_Y, axis=1)
                    mc_deltas[l] = dirichlet.maximize_delta(mc_Z, k)
                    c = dirichlet.forward_filter(mc_Y, mc_deltas[l], c0)
                else:
                    # Not modelling delta: use computed delta for everything
                    c = dirichlet.forward_filter(mc_Y, delta[i], c0)

                # Sum the mean parcel
                mc_estimates += dirichlet.backwards_mc_estimator(c, delta[i]) / M

            # Save results
            eta[:,i,:] = mc_estimates
            if model_delta:
                delta[i] = np.mean(mc_deltas)

        # Step 2: Maximize the cluster parameters

        W = []
        for j in range(k):
            theta[j], V, Wj, _ = dlm.dynamic_weighted_mle(Y, F_list[j], G_list[j], eta[:,:,j],
                                                          maxit=mnumit, numeps=numeps)
            phi[j,:] = 1 / np.diag(V)
            W.append(Wj)

    print('Done.')

    return eta, theta, phi, W, delta


class DynamicSamplerResult:
    '''
    Holds the results from a `sampler` run from the `dynamic` module.
    '''

    def __init__(self, numit, k, m, p, n, T):
        self.k = k

        self.theta = [np.empty((numit, T, p[j])) for j in range(k)]
        self.phi = [np.empty((numit, m)) for j in range(k)]
        self.W = [np.empty((numit, T, p[j], p[j])) for j in range(k)]

        self.Z = np.empty((numit, T, n), dtype=np.int64)
        self.eta = np.empty((numit, T, n, k))

        self.delta = np.empty((numit, n))

        self._max = numit
        self._curr = 0

    def include(self, theta, phi, W, Z, eta, delta):
        '''
        Include samples from a new iteration into the result.
        '''

        if self._curr == self._max:
            raise RuntimeError("Tried to include more samples than initially specified.")

        it = self._curr
        self._curr += 1

        for j in range(self.k):
            # NOTE: `phi` is passed as an ndarray but here it's a list of ndarrays
            # This is because in the `sampler` code it makes sense for the cluster
            # dimension to stay as the rows of the ndarray, but here, for consitency
            # with the other cluster-specific variables, the cluster dimension is
            # a list.

            self.theta[j][it,:,:] = theta[j]
            self.phi[j][it,:] = phi[j]
            self.W[j][it,:,:,:] = W[j]

        self.Z[it,:,:] = Z.argmax(axis=2)
        self.eta[it,:,:,:] = eta

        self.delta[it,:] = delta

    def means(self):
        """
        Return the estimates for the parameters based on means.
        """

        theta = [self.theta[j].mean(axis=0) for j in range(self.k)]
        phi = [self.phi[j].mean(axis=0) for j in range(self.k)]
        W = [self.W[j].mean(axis=0) for j in range(self.k)]

        Z = self.Z.mean(axis=0)
        eta = self.eta.mean(axis=0)
        delta = self.delta.mean(axis=0)

        return theta, phi, W, Z, eta, delta


def sampler(Y: np.ndarray, F_list: List[np.ndarray], G_list: List[np.ndarray], num_samples: int = 2000):
    '''
    Obtain samples from the parameter's posterior distributions based on
    Gibbs sampling and Forward Filtering and Backwards Sampling procedures.

    Args:
        Y:      A matrix with T rows and n*m columns.
        k:      Number of clusters.
        numit:  Number of samples to be generated.

    Returns:
        A DynamicSamplerResult object.
    '''

    # Initialize model parameters
    k, m, p, n, T, idx_map = common.get_dimensions(Y, F_list, G_list)
    _, theta, phi, eta = common.initialize(Y, F_list, G_list, dynamic=True)
    delta = np.ones(n) * 0.5
    Z = np.empty((T, n, k))  # First thing that happens is generating Z, so it doesn't have to be explicitly initialized.

    # Initialize the object to be returned
    chains = DynamicSamplerResult(num_samples, k, m, p, n, T)

    # Parameter for the prior distribution of each delta_i
    c0 = np.ones(k) * 0.1

    # Alias for the multivariate normal pdf
    f = sps.multivariate_normal.pdf

    #-- Gibbs sampling

    for it in tqdm(range(num_samples)):
        #-- Sampling from the indicator variables

        sigma = [np.diag(1. / phi[j]) for j in range(k)]

        for i in range(n):
            for t in range(T):
                densities = np.array([f(Y[t, i], np.dot(F_list[j], theta[j][t]), sigma[j]) for j in range(k)])
                weights = eta[t, i] * densities
                Z[t, i] = rng.multinomial(1, weights / weights.sum())

        #-- Sampling for the Dirichlet model discount factor

        for i in range(n):
            delta[i] = dirichlet.sample_delta(np.argmax(Z[:,i,:], axis=1), k)

        #-- Sampling the clustering weights for each observation

        for i in range(n):
            c = dirichlet.forward_filter(Z[:,i], delta[i], c0)
            eta[:,i] = dirichlet.backwards_sampler(c, delta[i])

        #-- Sampling DLM states and variances for each cluster

        for j in range(k):
            # Get which observations belong to the j-th cluster according to the dummies
            Y_j = []
            for t in range(T):
                included_obsevations_index = [i for i in range(n) if Z[t,i,j] == 1]
                included_observations_real_indexes = [indexes for i in included_obsevations_index for indexes in idx_map[i]]
                Y_j.append(Y[t, included_observations_real_indexes])

            # Perform appropriately sized tiling operations
            n_j = [len(y) for y in Y_j]
            F_j = [np.tile(F_list[j], (n, 1)) for n in n_j]
            V_j = [np.diag(np.tile(1.0 / phi[j], n)) for n in n_j]

            # Perform filtering and smoothing procedure
            a, R, M, C, W = dlm.filter_df_dyn(Y_j, F_j, G_list[j], V_j)
            M, C = dlm.smoother(G_list[j], a, R, M, C)

            # Sample from theta and phi
            obs_error = np.ones(m)
            for t in range(T):
                theta[j][t] = rng.multivariate_normal(M[t], C[t])
                obs_error += ((np.dot(F_j[t], theta[j][t]) - Y_j[t]) ** 2).reshape((n_j[t], m)).sum(axis=0)

            phi[j] = rng.gamma(np.sum(n_j) * T + 1, 1 / obs_error)

        # Save values
        chains.include(theta, phi, W, Z, eta, delta)

    # Return chains dropping burnin phase
    return chains
