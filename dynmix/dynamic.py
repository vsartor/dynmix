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

        self._max = numit
        self._curr = 0


    def include(self, theta, phi, W, Z, eta):
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
            # dimension to stay as the rows of the ndarray but here, for consitency
            # with the other cluster-specific variables, the cluster dimension is
            # a list.

            self.theta[j][it,:,:] = theta[j]
            self.phi[j][it,:] = phi[j]
            self.W[j][it,:,:,:] = W[j]
        
        self.Z[it,:,:] = Z
        self.eta[it,:,:,:] = eta


    def means(self):
        """
        Return the estimates for the parameters based on means.
        """

        theta = [self.theta[j].mean(axis=0) for j in range(self.k)]
        phi = [self.phi[j].mean(axis=0) for j in range(self.k)]
        W = [self.W[j].mean(axis=0) for j in range(self.k)]

        Z = self.Z.mean(axis=0)
        eta = self.eta.mean(axis=0)

        return theta, phi, W, Z, eta


def sampler(Y, F_list, G_list, delta, numit=2000):
    '''
    Uses the Gibbs sampler to obtain samples from the posterior for dynamic
    clusterization of n m-variate time-series, all observed throughout the
    same T time instants.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        numit: Number of iterations for the algorithm to run. Defaults to 2000.

    Returns:
        A `DynamicSamplerResult` object.
    '''

    #-- Preamble 

    k, m, p, n, T, index_map = common.get_dimensions(Y, F_list, G_list)
    _, theta, phi, eta = common.initialize(Y, F_list, G_list, dynamic=True)
    Z = common.compute_weights_dyn(Y, F_list, G_list, theta, phi, eta).argmax(axis=2)
    chains = DynamicSamplerResult(numit, k, m, p, n, T)

    c0 = np.ones(k) * 0.5

    # Allocate memory for W
    W = [np.empty((T, p[j], p[j])) for j in range(k)]

    #-- Gibbs sampler

    for it in range(numit):
        if it % 200 == 0:
            print(f'dynmix.dynamic.sampler [{it}|{numit}]')

        # Sample for Z
        for t in range(T):
            for i in range(n):
                Z[t,i] = rng.choice(k, p=eta[t,i])

        # Sample for eta
        for i in range(n):
            multi_Z = np.array([common.basis_vec(Z[t,i], k) for t in range(T)])
            c = dirichlet.forward_filter(multi_Z, delta[i], c0)
            eta[:,i,:] = dirichlet.backwards_sampler(c, delta[i])

        # Sample for theta, W, phi

        # REWRITE THIS BLOCK! HERE THE SIZE OF Y AT EACH TIME-STEP VARIES
        # USE FILTER_DF_DYN!!!
        for j in range(k):
            member_n = np.empty(T, dtype=np.int)
            member_Y = []

            for t in range(T):
                # Which observations are members of this cluster
                member_nat_idx = [i for i in range(n) if Z[t,i] == j]
                # Which are the Y indexes for the members of this cluster
                member_idx = [idx for i in member_nat_idx for idx in index_map[i]]

                member_n[t] = len(member_nat_idx)
                member_Y.append(Y[t,member_idx])

            G = G_list[j]
            F = [np.tile(F_list[j], (n, 1)) for n in member_n]
            V = [np.diag(np.tile(phi[j], n)) for n in member_n]

            a, R, M, C, W[j][:,:,:] = dlm.filter_df_dyn(member_Y, F, G, V)
            M, C = dlm.smoother(G, a, R, M, C)

            obs_error = np.ones(m)
            for t in range(T):
                theta[j][t] = rng.multivariate_normal(M[t], C[t])
                obs_error += ((np.dot(F[t], theta[j][t]) - member_Y[t])**2).reshape((member_n[t], m)).sum(axis=0)
            
            phi[j] = rng.gamma(member_n.sum() * T + 1, 1 / obs_error)

        # Save values
        chains.include(theta, phi, W, Z, eta)

    return chains
