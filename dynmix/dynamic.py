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
from . import static
from . import dirichlet


def delta(Y, F_list, G_list, W=5, seed=2):
    '''
    Selects a discount factor for each time-series based on static mixture
    of DLMs.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        W: Number of windows for the time-span to be divided over.
        seed: Seed used for reproducing similar classifications.

    Returns:
        The discount factor vector.
    '''

    state = rng.get_state()

    _, _, _, n, T, _ = common.get_dimensions(Y, F_list, G_list)
    time_windows = [range(int(i * T / W), int((i + 1) * T / W)) for i in range(W)]

    classifications = np.empty((W, n))

    for l, window in enumerate(time_windows):
        print(f'Window {l} out of {W}...')

        rng.seed(seed)
        eta, _, _, _ = static.estimator(Y[window], F_list, G_list, numit=10, mnumit=50)
        classifications[l] = eta.argmax(axis=1)

    scores = classifications.var(axis=0)

    rng.set_state(state)
    return 0.5 + 0.45 * (1 - (scores - scores.min()) / (scores.max() - scores.min()))


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
        
        self.Z[it,:,:] = Z.argmax(axis=2) #TODO: REMOVE argmax(axis=2) WHEN THE INPUT Z IS NOT MULTINOMIAL
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


def sampler(Y, F_list,  G_list, delta, numit=2000):
    '''
    Obtain samples from the posterior of a simple univariate
    first-order polynomial DLM dynamic clustering.

    Args:
        Y: A matrix with T rows and n*m columns.
        k: Number of clusters.
        delta: Discount factors to be used.
        numit: Number of iterations for the algorithm to run.
        
    Returns:
        A DynamicSamplerResult object.
    '''

    #-- Preamble 

    k, m, p, n, T, idx_map = common.get_dimensions(Y, F_list, G_list)
    _, theta, phi, eta = common.initialize(Y, F_list, G_list, dynamic=True)
    chains = DynamicSamplerResult(numit, k, m, p, n, T)

    Z = np.empty((T, n, k))

    f = sps.multivariate_normal.pdf

    c0 = np.ones(k) * 0.1

    # Gibbs iterations
    for it in range(numit):
        if it % 200 == 0:
            print(f'dynmix.dynamic.sampler [{it}|{numit}]')


        # Sample membership dummy parameters for each unit

        for i in range(n):
            for t in range(T):
                f_vals = np.array([f(Y[t, i], np.dot(F_list[j], theta[j][t]), np.diag(1. / phi[j]))
                                   for j in range(k)])
                weights = eta[t, i] * f_vals
                Z[t, i] = rng.multinomial(1, weights / weights.sum())


        # Sample Dirichlet states for each unit

        for i in range(n):
            c = dirichlet.forward_filter(Z[:,i], delta[i], c0)
            eta[:,i] = dirichlet.backwards_sampler(c, delta[i])


        # Sample DLM states and parameters for each cluster

        for j in range(k):
            YJ = []
            nJ = []
            
            for t in range(T):
                nat_idx = [i for i in range(n) if Z[t,i,j] == 1]
                idx = [idx for i in nat_idx for idx in idx_map[i]]
                YJ.append(Y[t, idx])
            
            nJ = [len(y) for y in YJ]
            FJ = [np.tile(F_list[j], (n, 1)) for n in nJ]
            VJ = [np.diag(np.tile(1.0 / phi[j], n)) for n in nJ]

            a, R, M, C, W = dlm.filter_df_dyn(YJ, FJ, G_list[j], VJ)
            M, C = dlm.smoother(G_list[j], a, R, M, C)

            obs_error = np.ones(m)
            for t in range(T):
                theta[j][t] = rng.multivariate_normal(M[t], C[t])
                obs_error += ((np.dot(FJ[t], theta[j][t]) - YJ[t])**2).reshape((nJ[t], m)).sum(axis=0)

            phi[j] = rng.gamma(np.sum(nJ) * T + 1, 1 / obs_error)

        # Save values
        chains.include(theta, phi, W, Z, eta)

    # Return chains dropping burnin phase
    return chains
