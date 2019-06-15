'''
This module implements common utility functions for the static, independent
and dynamic modules.

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


def get_dimensions(Y, F_list, G_list):
    '''
    Returns the problem dimensions given the cannonical arguments.

    Args:
        Y: A matrix with T rows and n*m columns.
        F_list: A list with k specifications for the F matrix of each cluster.
        G_list: A list with k specifications for the G matrix of each cluster.

    Returns:
        k: The number of clusters.
        m: The dimension of a single observation.
        p: List with the dimension of state space for each cluster.
        n: Number of replicates.
        T: Size of time window.
        index_mask: A map from the observation index to the corresponding Y indexes.
    '''

    # The number of clusters
    k = len(F_list)

    # The dimension of a single observation is m, and is given by F
    # Note that all F should have the same number of rows, since only
    # state dimension can vary from cluster to cluster
    m = F_list[0].shape[0]

    # The dimension of the states for each cluster is given by the number
    # of columns in each F
    p = [F.shape[1] for F in F_list]

    # The number of replicates is n. Since the number of columns for y is
    # n*m we need only to divide and perform a small check to make sure
    # everything is ok.
    n = Y.shape[1] / m
    if n != int(n):
        raise ValueError('Bad dimensions n and m')
    n = int(n)

    # The number of time instants is T and is given by the rows of Y
    T = Y.shape[0]

    # Create a map associating the observation index with indexes in Y
    index_mask = {i: range(i * m, (i + 1) * m) for i in range(n)}

    return k, m, p, n, T, index_mask


def compute_weights_dyn(Y, F_list, G_list, theta, phi, eta=None):
    '''
    Compute the membership weights of the mixture model. This is essentially a
    function for the result of the E-step of the dynamic mixture of DLMs model.

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

    k, _, _, n, T, idxmap = get_dimensions(Y, F_list, G_list)
    weights = np.empty((T, n, k))

    if eta is None:
        # Disconsider eta from the computation
        eta = np.ones((T, n, k))

    #-- Algorithm

    for t in range(T):
        for i in range(n):
            for j in range(k):
                F = F_list[j]
                thetaj = theta[j]
                V = np.diag(1 / phi[j])
                weights[t,i,j] = sps.multivariate_normal.pdf(Y[t,idxmap[i]], np.dot(F, thetaj[t]), V)
            weights[t,i] /= weights[t,i].sum()

    return weights


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

    k, _, _, n, T, idxmap = get_dimensions(Y, F_list, G_list)
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

        include_mask = np.ones(k, dtype=np.bool)

        # Compute the pdfs
        for j in range(k):
            F = F_list[j]
            thetaj = theta[j]
            V = np.diag(1 / phi[j])
            for t in range(T):
                pdfs[j,t] = sps.multivariate_normal.pdf(Y[t,obs_idx], np.dot(F, thetaj[t]), V)
    
            # Zeros become minimum non-zero computed value
            zero_mask = pdfs[j] == 0
            if zero_mask.all():
                include_mask[j] = False
            elif zero_mask.any():
                pdfs[j][zero_mask] = np.min(pdfs[j][np.invert(zero_mask)])
    
        # Compute the mean order of magnitude for each cluster and fetch the maximum.
        if np.all(np.invert(include_mask)):
            weights[i] = 1 / k
            continue
        O_bar = np.mean(np.log10(pdfs[include_mask]), axis=1).max()

        # Adjust the order of magnitude
        pdfs *= 10**(-O_bar)

        # Perform usual computations based on the entries of pdfs
        for j in range(k):
            weights[i,j] = eta[i,j] * pdfs[j].prod()
        weights[i] /= weights[i].sum()

    return weights


def initialize(Y, F_list, G_list, dynamic):
    '''
    Uses an adaptation of the kmeans++ to initialize the model parameters for
    the mixture of DLMs.

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

    k, m, p, n, T, index_mask = get_dimensions(Y, F_list, G_list)

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
        phi[j,:] = 1 / np.diag(V_est)

    # Step 4: Compute the membership parameters
    if dynamic:
        eta = compute_weights_dyn(Y, F_list, G_list, theta, phi)
    else:
        eta = compute_weights(Y, F_list, G_list, theta, phi)

    return centroids, theta, phi, eta


def basis_vec(i, k):
    '''
    Returns the i-th canonical vector for the k-dimensional euclidean space.

    Args:
        i: Index of basis vector.
        k: Euclidean space dimension.

    Returns:
        The i-th canonical vector for the k-dimensional euclidean space
    '''

    vec = np.zeros(k)
    vec[i] = 1.0
    return vec
