'''
This module implements static mixture estimation through the Expectation
Maximization algorithm. It's a simple atemporal clustering technique
included for completeness.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import numpy.random as npr

import scipy.stats as sps


def gamma(x, theta, eta):
    '''
    Conditional probability of X being a Normal(theta_j) weighted by eta.

    Args:
        x: p-variate observation.
        theta: List with k pairs. Each pair should have a p-variate mean vector
               and a p-order covariance matrix, respectively.
        eta: Array of k weights that sum to one.

    Returns:
        Array with the probabilities.
    '''

    weight = np.fromiter([eta[j] * sps.multivariate_normal.pdf(x, *theta[j])
                          for j in range(eta.size)], float)

    return weight / weight.sum()


def gmm(X, k, init_labels=None, numit=100, unique=False, show_it=False):
    '''
    Estimates theta and eta based on EM algorithm.

    Args:
        X: Array with a p-variate observation in each row.
        k: Number of clusters to be considered.
        init_labels: The indexes of the observations to be used as means
                     while initializing the clustering algorithm. If None
                     is given, random choices are made.
        numit: Number of iterations for the EM algorithm.
        unique: Boolean indicating whether different weights should be
                estimated for each observation or if all of them come
                from the same mixture.
        show_it: Update progress every `show_it` iterations. If

    Returns:
        eta: If unique is False, a vector containing the mixture weights
             for the population. If unique is True, contains a the mixture
             weights for each of the observations.
        theta: The list of parameters estimated for each cluster.
    '''

    n, p = X.shape

    # Initialize means from random observations, variances deterministically
    # and weights from Dirichlet.

    if init_labels is None:
        init_labels = npr.choice(n, k, False)
    theta = [(X[init_labels[j]], np.eye(p) * 10) for j in range(k)]

    if unique:
        eta = npr.dirichlet(np.ones(k) / k)
    else:
        eta = [npr.dirichlet(np.ones(k) / k) for i in range(n)]

    for l in range(numit):
        if show_it and l % show_it == 0:
            print(f'GMM-EM {l}/{numit}')

        # E-step: calculate weights

        if unique:
            gammas = np.array(list([gamma(X[i], theta, eta)
                                    for i in range(n)]))
        else:
            gammas = np.array(list([gamma(X[i], theta, eta[i])
                                    for i in range(n)]))

        # M-step: calculate new values for parameters

        eta = gammas.mean(axis=0) if unique else gammas

        for j in range(k):
            gamma_vec = gammas[:, j]
            gamma_n = gamma_vec.sum()

            mu = np.sum(X * gamma_vec[:, np.newaxis], axis=0) / gamma_n

            sigma = np.zeros((p, p))
            for i in range(n):
                # NOTE: atleast_2d transforms the 1d array into a line matrix.
                #       As such, z.T is a column vector and z is a row vector.
                z = np.atleast_2d(X[i] - mu)
                sigma += gamma_vec[i] * np.dot(z.T, z)
            sigma /= gamma_n

            theta[j] = (mu, sigma)

    return eta, theta
