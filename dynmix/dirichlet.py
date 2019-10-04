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
import numpy.random as rng

from numba import njit


@njit
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

    n, k = Y.shape
    c = np.empty((n, k))

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

    n = c.shape[0]
    eta = np.empty(c.shape)

    eta[n-1] = rng.dirichlet(c[n-1], 1)[0]

    for t in range(n-1, 0, -1):
        csum = c[t-1].sum()
        S = rng.beta(delta * csum, (1 - delta) * csum)
        u = rng.dirichlet((1 - delta) * c[t-1], 1)[0]
        eta[t-1] = S * eta[t] + (1 - S) * u

    return eta


def backwards_mc_estimator(c: np.ndarray, delta: float, resolution: int = 10, M: int = 50):
    '''
    Estimates the mode through Monte-Carlo simulation.
    '''

    T, k = c.shape

    grid = np.linspace(0, 1, resolution + 1)
    grid_lo = grid[:-1]
    grid_hi = grid[1:]
    grid_mu = 0.5 * (grid_hi + grid_lo)

    bin_counts = np.zeros((T, k, resolution))

    # Simulate M times, and add a point to the bins that got the value
    for _ in range(M):
        mc_sample = backwards_sampler(c, delta)
        for t in range(T):
            for j in range(k-1):
                index = np.argmax(np.logical_and(grid_lo < mc_sample[t,j], mc_sample[t,j] < grid_hi))
                bin_counts[t, j, index] += 1

    # Get the bins with the highest count
    eta = np.empty((T, k))
    for t in range(T):
        for j in range(k-1):
            eta[t,j] = grid_mu[np.argmax(bin_counts[t, j])]
        # Last cluster's weights is one minus the sum of the other columns
        eta[t,k-1] = 1 - eta[t,:-1].sum()

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


def dirichlet_mode_core(c):
    return (c - 1) / (c.sum() - c.size)


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

    bad_mask = c < 1
    if np.all(bad_mask):
        return mod_dirichlet_mode(c + 0.1, a, b)
    elif np.any(bad_mask):
        # Recompute for c > 1, and consider a zero for the ones where c < 0
        good_mask = np.invert(bad_mask)
        good_mode = dirichlet_mode_core(c[good_mask])
        dirichlet_mode = np.zeros(c.size)
        dirichlet_mode[good_mask] = good_mode
    else:
        dirichlet_mode = dirichlet_mode_core(c)

    # Mod-Dirichlet transformation
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
        return mod_dirichlet_parameters(c + 0.1, delta, eta)
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


def backwards_estimator(c: np.ndarray, delta: float) -> np.ndarray:
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
    c0_one = 0.01
    c0_sum = c0_one * k

    for t in range(1, T+1):
        numerator = delta ** (t-1) * c0_one
        for l in range(0, t-1):
            if Z[t-2-l] - Z[t-1]:
                numerator += delta ** l

        denominator = delta ** (t-1) * c0_sum + (1 - delta ** (t-1)) / (1 - delta)
        log_marginal += np.log(numerator) - np.log(denominator)

    return log_marginal


def sample_delta(Z: np.array, k: int, resolution: int = 50) -> float:
    '''
    Obtains a sample from the marginal of the Dirichlet Model's discount factor.

    Z:          Vector of integers between 0 and k - 1.
    k:          Integer value higher than 0.
    resolution: Grid resolution for delta candidates. Integer value higher than 10.
    '''

    #-- Uses a Sampling Importance Resampling approach for simulating the values:
    #--     1. Generates some samples from a candidate distribution
    #--     2. Computes a weight which is based on the likelihood of each point
    #--     3. Resample the original samples from step 1, weighted by the values from step 2

    delta_grid = rng.uniform(0.05, 0.95, size=resolution)
    likelihood = np.exp(delta_marginal(delta_grid, Z, k))

    if np.all(likelihood == 0):
        # The candidates are all numerically impossible, rerun the routine
        return sample_delta(Z, k, resolution)

    return rng.choice(delta_grid, 1, p=likelihood / likelihood.sum())[0]


def maximize_delta(Z: np.array, k: int, resolution: int = 10) -> float:
    '''
    Obtains the maximum from the marginal of the Dirichlet Model's discount factor.

    Z:          Vector of integers between 0 and k - 1.
    k:          Integer value higher than 0.
    resolution: Grid resolution for delta candidates. Integer value higher than 10.
    '''

    # Simple grid-based discrete optimization.

    delta_grid = np.linspace(0.01, 0.99, resolution)
    marginal_likelihood = delta_marginal(delta_grid, Z, k)
    return delta_grid[np.argmax(marginal_likelihood)]
