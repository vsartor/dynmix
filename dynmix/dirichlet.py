'''
This module implements Dirichlet process related algorithms from
Fonseca & Ferreira (2017), allowing implementation of FFBS algorithms
for Monte Carlo simulations.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import numpy.random as npr


def forward_filter(Y, delta, c0):
    '''
    Performs forward filtering algorithm for a Dirichlet process as
    proposed in Fonseca & Ferreira (2017).

    Args:
        Y: The matrix of observations from a multinomial distribution.
        delta: The discount factor.
        c0: The prior knowledge about the Dirichlet process.

    Returns:
        c: The matrix containing parameters of the online distribution
        of the Dirichlet states.
    '''

    #-- Preamble

    n, k = Y.shape
    c = np.empty((n, k))

    #-- Algorithm

    c[0] = delta * c0 + Y[0]
    for t in range(1, n):
        c[t] = delta * c[t-1] + Y[t]

    return c


def backwards_sampler(c, delta):
    '''
    Performs backwards sampling algorithm for a Dirichlet process as
    proposed in Fonseca & Ferreira (2017).

    Args:
        c: The resulting matrix from foward filtering.
        delta: The discount factor.

    Returns:
        eta: A sample from the posterior distribution.
    '''

    #-- Preamble

    n = c.shape[0]
    eta = np.empty(c.shape)

    #-- Algorithm
    eta[n-1] = npr.dirichlet(c[n-1], 1)[0]

    for t in range(n-1, 0, -1):
        csum = c[t-1].sum()
        S = npr.beta(delta * csum, (1 - delta) * csum)
        u = npr.dirichlet((1 - delta) * c[t-1], 1)[0]
        eta[t-1] = S * eta[t] + (1 - S) * u

    return eta


def mod_dirichlet_mean(c, a, b):
    '''
    Returns the mean of a Mod-Dirichlet distribution from Appendix A.

    Args:
        c: The Dirichlet parameter.
        a: The minimum parameter.
        b: The maximum parameter.

    Returns:
        The mean vector.
    '''

    return (b - a) * c / c.sum() + a


def mod_dirichlet_mode(c, a, b):
    '''
    Returns the mode of a Mod-Dirichlet distribution from Appendix A.

    Args:
        c: The Dirichlet parameter.
        a: The minimum parameter.
        b: The maximum parameter.

    Returns:
        The mean vector.
    '''

    if np.any(c < 1):
        raise RuntimeError('Invalid dirichlet mode')

    dirichlet_mode = (c - 1) / (c.sum() - c.size)
    return (b - a) * dirichlet_mode + a


def mod_dirichlet_parameters(c, delta, eta):
    '''
    Returns the parameter set for the Mod-Dirichlet from
    Proposition 3.2.

    Args:
        c: The foward filtering result for the time of interest.
        delta: Discount factor.
        eta: The mode from one time instant ahead.

    Returns:
        The parameter set c, a, b.
    '''

    k = eta.size

    # Step 1. Calculate the mode for S

    c_sum = c.sum()
    alpha = delta * c_sum
    beta = (1 - delta) * c_sum

    if alpha < 1 and beta < 1:
        raise RuntimeError('Invalid mode for S')
    elif alpha <= 1:
        s = 0
    elif beta <= 1:
        s = 1
    else:
        s = (alpha - 1) / (alpha + beta - 2)

    # Step 2. Calculate the arguments as a function of the mode for S

    c = (1 - delta) * c
    a = s * eta
    b = (1 - s) * np.ones(k) + s * eta

    return c, a, b


def backwards_estimator(c, delta):
    '''
    Instead of performing usual backwards sampling, it iterates
    backwards picking values for omega according to the mode of
    the distribution.

    Args:
        c: The resulting matrix from forward filtering.
        delta: The discount factor.

    Returns:
        eta: A matrix containing modes from the posterior distribution.
    '''

    n = c.shape[0]
    eta = np.empty(c.shape)

    # For the last eta's distribution we can do it directly since
    # it's a known Dirichlet(c[n-1]), and writtn as a Mod-Dirichlet
    # it is Mod-Dirichlet(c[n-1], 0, 1).

    eta[n-1] = mod_dirichlet_mean(c[n-1], 0, 1)

    # For the other ones we need to find the Mod-Dirichlet parameters
    # conditional the previous mode being the real value.

    for t in range(n-2, -1, -1):
        params = mod_dirichlet_parameters(c[t], delta, eta[t+1])
        eta[t] = mod_dirichlet_mean(*params)

    return eta


def delta_marginal(delta: float, Z: np.array, k: int) -> float:
    '''
    The marginal of delta given the multinomial observations. Specifically crafted
    for the case of Z_t | eta_t ~ Multinom(1, eta_t). The prior is hard-coded
    as uninformative.

    delta:  Real value between 0 and 1.
    Z:      NumPy vector of integers between 0 and k - 1.
    k:      Integer value higher than 0.
    '''

    T = len(Z)

    log_marginal = 0
    c0_one = 0.5
    c0_sum = c0_one * k

    for t in range(1, T+1):
        numerator = delta ** (t-1) * c0_one
        for l in range(0, t-1):
            if Z[t-2-l] - Z[t-1]:
                numerator += delta ** l

        denominator = delta ** (t-1) * c0_sum + (1 - delta ** (t-1)) / (1 - delta)
        log_marginal += np.log(numerator) - np.log(denominator)

    return log_marginal


def maximize_delta(Z: np.array, k: int, resolution: int = 40, mean: bool = False):
    '''
    Z:          NumPy vector of integers between 0 and k - 1.
    k:          Integer value higher than 0.
    resolution: Grid resolution for delta candidates. Integer value higher than 10.
    '''

    delta_grid = np.linspace(0.01, 0.99, resolution)
    marginal_likelihood = np.exp(delta_marginal(delta_grid, Z, k))
    return delta_grid[np.argmax(marginal_likelihood)] if not mean else np.sum(delta_grid * marginal_likelihood) / np.sum(marginal_likelihood)
