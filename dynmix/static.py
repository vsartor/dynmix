'''
This module implements time-series mixture model for clustering using
DLMs for each of the k clusters.

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


def compute_weights(Y, F_list, G_list, theta, phi, eta=None):
    '''
    Compute the membership weights of the mixture model. This is essentially a
    function for the result of the E-step of the static mixture of DLMs model.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        theta: The current estimates of theta.
        phi: The current estimates of phi.
        eta: The current estimates of eta. If not passed, disconsiders it.

    Returns:
        weights: The weights array.
    '''

    #-- Preamble

    k, _, _, n, T, idxmap = common.get_dimensions(Y, F_list, G_list)
    weights = np.empty((n, k))

    if eta is None:
        # Disconsider eta from the computation
        eta = np.ones((n, k))

    #-- Algorithm

    # For each observation, compute the weights
    for i in range(n):
        obs_idx = idxmap[i]

        # Allocate space for each pdf result to be computed
        pdfs = np.empty((k, T))
        # Include mask
        include_mask = np.ones(k, dtype=np.bool)

        # Compute the pdfs
        for j in range(k):
            F = F_list[j]
            thetaj = theta[j]
            V = np.diag(1 / phi[j])
            for t in range(T):
                pdfs[j,t] = sps.multivariate_normal.pdf(Y[t,obs_idx], np.dot(F, thetaj[t]), V)
                if pdfs[j,t] == 0:
                    include_mask[j] = False
                    pdfs[j,:] = 0 # Enter a dummy value
                    break
    
        # Compute the mean order of magnitude for each cluster and fetch the maximum.
        O_bar = np.mean(np.log10(pdfs[include_mask]), axis=1).max()

        # Adjust the order of magnitude
        pdfs *= 10**(-O_bar)

        # Perform usual computations based on the entries of pdfs
        for j in range(k):
            weights[i,j] = eta[i,j] * pdfs[j].prod()
        weights[i] /= weights[i].sum()

    return weights


def initialize(Y, F_list, G_list):
    '''
    Uses an adaptation of the kmeans++ to initialize the model parameters for
    the static mixture of DLMs.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.

    Returns:
        centroids: The indexes of the chosen representatives for each cluster.
        theta: A list with the theta for each cluster.
        phi: A list with the phi for each cluster.
        eta: A list with the eta for each time-series.
    '''

    #-- Initialization

    k, m, p, n, T, index_mask = common.get_dimensions(Y, F_list, G_list)

    # Allocate space for parameters
    theta = [np.empty((T, p[j])) for j in range(k)]
    phi = np.empty((k, m))

    #-- Algorithm

    # Step 0: Initialize algorithm-specific variables
    centroids = []
    candidates = [i for i in range(n)]
    distances = []

    # Step 1: Pick an observation at random
    centroids.append(rng.choice(candidates, 1)[0])

    # Step 2: Pick further centroids with higher probability of picking
    # one further away from the already existing ones
    for _ in range(k-1):
        # Remove from list of candidates the last chosen centroid
        candidates.remove(centroids[-1])

        # Add to `distances` the distance between every observation and the new centroid
        for j, centroid_index in enumerate(centroids):
            centroid = Y[:,index_mask[centroid_index]]
            distances.append([np.sum((Y[:,index_mask[i]] - centroid)**2) for i in range(n)])

        # NOTE: I do this not to repeat the distance computation every step for the same
        # centroids, the (tiny) drawback is computing for every observation instead of
        # every candidate. This is tiny because there should be a small number of centroids
        # and thus a small number of observations for which the computation is being performed
        # unecessarily. The reshuffling necessary to eliminate already chosen candidates from
        # the distances matrix would incur a much higher cost.

        # Get the biggest distance for each observation and only get this for candidates
        weights = np.max(np.array(distances), axis=0)[candidates]
        
        # Draw a new centroid weighted by the distance
        probs = weights / weights.sum()
        centroids.append(rng.choice(candidates, 1, p=probs)[0])
    
    # Step 3: Now that centroid observations have been picked, initialize the
    # cluster parameters based on MLE estimation which is based purely on them.
    
    # TODO: Current ordering is assumed to be arbitrary, which is only true if all
    # F_j and G_j are the same. When it isn't adjust all k models for all k centroids
    # and pick the highest likelihood candidate for each model.

    for j in range(k):
        theta_est, V_est, _, _ = dlm.mle(Y[:,index_mask[centroids[j]]], F_list[j], G_list[j])

        theta[j][:,:] = theta_est
        phi[j,:] = np.diag(V_est)

    # Step 4: Compute the membership parameters
    eta = compute_weights(Y, F_list, G_list, theta, phi)

    return centroids, theta, phi, eta


def estimator(Y, F_list, G_list, numit=20, mnumit=100, numeps=1e-6):
    '''
    Uses Expectation-Maximization to estimate statist clusterization of n
    m-variate time-series, all observed throughout the same T time instants.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        numit: Number of iterations for the algorithm to run.
        mnumit: Maximum number of iterations for the M-step algorithm to run.
        numeps: Numerical precision for the M-step algorithm.

    Returns:
        eta: A list with the eta for each time-series.
        theta: A list with the theta for each cluster.
        phi: A list with the phi for each cluster.
        W: A list with the W for each cluster.
    '''

    #-- Preamble 

    k, _, p, _, T, _ = common.get_dimensions(Y, F_list, G_list)
    _, theta, phi, eta = initialize(Y, F_list, G_list)

    #-- Algorithm

    # NOTE: Because W is a byproduct of the df_filter and not directly
    # estimated, it will only be saved after all estimation has finished
    # not to waste time and space storing something the routine is not using.

    for _ in range(numit):
        # NOTE: There is no need for an E-step where weights = compute_weights
        # and then in the M-step eta = weights. Just set eta = compute_weights.

        eta = compute_weights(Y, F_list, G_list, theta, phi, eta)
        
        for j in range(k):
            theta[j], V, _, _ = dlm.weighted_mle(Y, F_list[j], G_list[j], eta[:,j],
                                                 maxit=mnumit, numeps=numeps)
            phi[j,:] = 1 / np.diag(V)

    # NOTE: Now compute the W in a last-step

    W = [np.empty((T, p[j])) for j in range(k)]

    eta = compute_weights(Y, F_list, G_list, theta, phi, eta)
    for j in range(k):
        theta[j], V, W[j], _ = dlm.weighted_mle(Y, F_list[j], G_list[j], eta[:,j],
                                                maxit=mnumit, numeps=numeps)
        phi[j,:] = 1 / np.diag(V)

    return eta, theta, phi, W


class StaticSamplerResult:
    '''
    Holds the results from a `sampler` run from the `static` module.
    '''

    def __init__(self, numit, k, m, p, n, T):
        self.k = k

        self.theta = [np.empty((numit, T, p[j])) for j in range(k)]
        self.phi = [np.empty((numit, m)) for j in range(k)]
        self.W = [np.empty((numit, T, p[j], p[j])) for j in range(k)]
        
        self.Z = np.empty((numit, n), dtype=np.int64)
        self.eta = np.empty((numit, n, k))

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
        
        self.Z[it,:] = Z
        self.eta[it,:,:] = eta


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


def _basis_vec(i, k):
    vec = np.zeros(k)
    vec[i] = 1.0
    return vec


def sampler(Y, F_list, G_list, numit=2000, ord_time=0):
    '''
    Uses the Gibbs sampler to obtain samples from the posterior for static
    clusterization of n m-variate time-series, all observed throughout the
    same T time instants.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.
        numit: Number of iterations for the algorithm to run. Defaults to 2000.
        ord_time: The time-index used for ordering. Defaults to 0.

    Returns:
        A `StaticSamplerResult` object.
    '''

    #-- Preamble 

    k, m, p, n, T, index_map = common.get_dimensions(Y, F_list, G_list)
    _, theta, phi, eta = initialize(Y, F_list, G_list)
    Z = compute_weights(Y, F_list, G_list, theta, phi, eta).argmax(axis=1)
    chains = StaticSamplerResult(numit, k, m, p, n, T)

    # Make sure clusters start with correct ordering
    order = np.argsort([theta_[ord_time,0] for theta_ in theta])

    theta = [theta[i] for i in order]
    phi = phi[order]

    # Allocate memory for W
    W = [np.empty((T, p[j], p[j])) for j in range(k)]

    #-- Gibbs sampler

    for it in range(numit):
        if it % 200 == 0:
            print(f'dynmix.static.sampler [{it}|{numit}]')

        # Sample for Z, eta
        weights = compute_weights(Y, F_list, G_list, theta, phi, eta)
        for i in range(n):
            Z[i] = rng.choice(k, p=weights[i])
            eta[i] = rng.dirichlet(np.ones(k) * 0.1 + _basis_vec(Z[i], k))

        # Should never happen but good sanity check.
        if len(np.unique(Z)) != k:
            raise RuntimeError("Empty clusters.")

        # Sample for theta, W, phi
        for j in range(k):
            member_indexes = [idx for i in range(n) for idx in index_map[i] if Z[i] == j]
            member_Y = Y[:,member_indexes]
            member_n = np.sum(Z == j)

            G = G_list[j]
            F = np.tile(F_list[j], (member_n, 1))
            V = np.diag(np.tile(1.0 / phi[j], member_n))

            a, R, _, _, M, C, W[j][:,:,:] = dlm.filter_df(member_Y, F, G, V)
            M, C = dlm.smoother(G, a, R, M, C)

            obs_error = np.ones(m)
            for t in range(T):
                theta[j][t] = rng.multivariate_normal(M[t], C[t])
                obs_error += ((np.dot(F, theta[j][t]) - member_Y[t])**2).reshape((member_n, m)).sum(axis=0)
            
            phi[j] = rng.gamma(member_n * T + 1, 1 / obs_error)

        # Impose ordering restriction

        order = np.argsort([theta_[ord_time,0] for theta_ in theta])
        
        theta = [theta[i] for i in order]
        phi = phi[order]

        # Save values
        chains.include(theta, phi, W, Z, eta)

    return chains