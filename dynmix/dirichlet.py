'''
Dirichlet process related algorithms from Fonseca & Ferreira (2017).

Copyright (c) Victhor S. Sartório. All rights reserved.
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np
import numpy.random as npr


def dirichlet_forward_filter(y, delta, c0):
    '''
    Performs forward filtering algorithm for a Dirichlet process as
    proposed in Fonseca & Ferreira (2017).
    '''
    n, k = y.shape
    c = np.empty((n, k))
    c[0] = delta * c0 + y[0]
    for t in range(1, n):
        c[t] = delta * c[t-1] + y[t]
    return c


def dirichlet_backwards_sampler(c, delta):
    '''
    Performs backwards sampling algorithm for a Dirichlet process as
    proposed in Fonseca & Ferreira (2017).
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