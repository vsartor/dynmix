'''
This module implements time-series mixture model for clustering using
DLMs for each of the k clusters where each time-series has a different,
independent membership at each time t.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import numpy.random as rng

from . import dlm
from . import common


def estimator(Y, F_list, G_list, numit=20, mnumit=100, numeps=1e-6):
    '''
    Uses Expectation-Maximization to estimate independent clusterization of n
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
    _, theta, phi, eta = common.initialize(Y, F_list, G_list, dynamic=True)

    #-- Algorithm

    # NOTE: Because W is a byproduct of the df_filter and not directly
    # estimated, it will only be saved after all estimation has finished
    # not to waste time and space storing something the routine is not using.

    for _ in range(numit):
        # NOTE: There is no need for an E-step where weights = compute_weights
        # and then in the M-step eta = weights. Just set eta = compute_weights.

        eta = common.compute_weights_dyn(Y, F_list, G_list, theta, phi, eta)
        
        for j in range(k):
            theta[j], V, _, _ = dlm.dynamic_weighted_mle(Y, F_list[j], G_list[j], eta[:,:,j],
                                                         maxit=mnumit, numeps=numeps)
            phi[j,:] = 1 / np.diag(V)

    # NOTE: Now compute the W in a last-step

    W = [np.empty((T, p[j])) for j in range(k)]

    eta = common.compute_weights_dyn(Y, F_list, G_list, theta, phi, eta)
    for j in range(k):
        theta[j], V, W[j], _ = dlm.dynamic_weighted_mle(Y, F_list[j], G_list[j], eta[:,:,j],
                                                        maxit=mnumit, numeps=numeps)
        phi[j,:] = 1 / np.diag(V)

    return eta, theta, phi, W
