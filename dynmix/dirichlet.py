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


def forward_filter(y, delta, c0):
    '''
    Performs forward filtering algorithm for a Dirichlet process as
    proposed in Fonseca & Ferreira (2017).

    Args:
        y: The matrix of observations from a multinomial distribution.
        delta: The discount factor for the model.
        c0: The prior knowledge about the Dirichlet process.

    Returns:
        The matrix c containing parameters of the online distribution
        of the Dirichlet states.
    '''
    n, k = y.shape
    c = np.empty((n, k))
    c[0] = delta * c0 + y[0]
    for t in range(1, n):
        c[t] = delta * c[t-1] + y[t]
    return c


def backwards_sampler(c, delta):
    '''
    Performs backwards sampling algorithm for a Dirichlet process as
    proposed in Fonseca & Ferreira (2017).

    Args:
        c: The online distribution parameters obtained from forward
           filtering.
        delta: The discount factor for the model.

    Returns:
        A matrix containing samples from the posterior distribution
        of the Dirichlet states.
    '''
    n = c.shape[0]
    omega = np.empty(c.shape)
    omega[n-1] = npr.dirichlet(c[n-1], 1)[0]
    for t in range(n-1, 0, -1):
        csum = c[t-1].sum()
        S = npr.beta(delta * csum, (1 - delta) * csum)
        u = npr.dirichlet((1 - delta) * c[t-1], 1)[0]
        omega[t-1] = S * omega[t] + (1 - S) * u
    return omega


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


def mod_dirichlet_parameters(c, delta, omega):
    '''
    Returns the parameter set for the Mod-Dirichlet from
    Proposition 3.2. Internal helper function.

    Args:
        c: The information parameter for the time of interest.
        delta: Discount factor for the model.
        omega: The mode from one time instant ahead.
    '''

    k = omega.size

    # Step 1. Calculate the mode for S

    c_sum = c.sum()
    alpha = delta * c_sum
    beta = (1 - delta) * c_sum
    s = alpha / (alpha + beta)

    # Step 2. Calculate the arguments as a function of the mean for S

    c = (1 - delta) * c
    a = s * omega
    b = (1 - s) * np.ones(k) + s * omega

    return c, a, b


def backwards_estimator(c, delta):
    '''
    Instead of performing usual backwards sampling, it iterates
    backwards picking values for omega according to the mean of
    the distribution, i.e. it takes the means instead of generating
    samples.

    Args:
        c: The online distribution parameters obtained from forward
           filtering.
        delta: The discount factor for the model.

    Returns:
        A matrix containing means from the posterior distribution
        of the Dirichlet states.
    '''

    n = c.shape[0]
    omega = np.empty(c.shape)

    # For the last omega's distribution we can do it directly since
    # it's a known Dirichlet(c[n-1]), and writtn as a Mod-Dirichlet
    # it is Mod-Dirichlet(c[n-1], 0, 1).

    omega[n-1] = mod_dirichlet_mean(c[n-1], 0, 1)

    # For the other ones we need to find the Mod-Dirichlet parameters
    # conditional the previous mode being the real value.

    for t in range(n-2, -1, -1):
        params = mod_dirichlet_parameters(c[t], delta, omega[t+1])
        omega[t] = mod_dirichlet_mean(*params)

    return omega
