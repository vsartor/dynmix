'''
This module implements simple Dynamic Linear Model routines while
also including routines a special case of "univariate" DLMs that
have a varying number `n_t` of repeated observations of `y_t` at
each time instant.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np


def filter(Y, F, G, V, W, m0=None, C0=None):
    '''
    Peforms the basic Kalman filter for a Dynamic Linear Model.

    Args:
        Y: The matrix of observations, with an observation in each row.
        F: The observational matrix.
        G: The evolutional matrix.
        V: The observational error covariance matrix.
        W: The evolutional error covariance matrix.
        m0: The prior mean for the states. Defaults to zeros.
        C0: The prior covariance for the states. Defaults to a diagonal
            matrix with entries equal to 10**6.

    Returns:
        Six arrays - the prior means and covariances, a and R, the one
        step ahead forecast means and covariances, f and R, and the
        online means and covariances, m and C.
    '''

    T = Y.shape[0]
    n, p = F.shape
    Gt = G.T
    Ft = F.T

    if Y.shape[1] != n or V.shape[0] != n or V.shape[1] != n:
        raise ValueError("Observational dimension mismatch")
    if G.shape[0] != p or G.shape[1] != p:
        raise ValueError("G dimension mismatch")
    if W.shape[0] != p or W.shape[1] != p:
        raise ValueError("W dimension mismatch")

    if m0 is None:
        m0 = np.zeros(p)
    if C0 is None:
        C0 = np.diag(np.ones(p)) * 10**6

    a = np.empty((T, p))
    R = np.empty((T, p, p))
    f = np.empty((T, n))
    Q = np.empty((T, n, n))
    m = np.empty((T, p))
    C = np.empty((T, p, p))

    a[0] = np.dot(G, m0)
    R[0] = np.dot(np.dot(G, C0), Gt) + W
    f[0] = np.dot(F, a[0])
    Q[0] = np.dot(np.dot(F, R[0]), Ft) + V
    e = Y[0] - f[0]
    Qinv = np.linalg.inv(Q[0])
    A = np.dot(np.dot(R[0], Ft), Qinv)
    m[0] = a[0] + np.dot(A, e)
    C[0] = R[0] - np.dot(np.dot(A, Q[0]), A.T)

    for t in range(1, T):
        a[t] = np.dot(G, m[t-1])
        R[t] = np.dot(np.dot(G, C[t-1]), Gt) + W
        f[t] = np.dot(F, a[t])
        Q[t] = np.dot(np.dot(F, R[t]), Ft) + V
        e = Y[t] - f[t]
        Qinv = np.linalg.inv(Q[t])
        A = np.dot(np.dot(R[t], Ft), Qinv)
        m[t] = a[t] + np.dot(A, e)
        C[t] = R[t] - np.dot(np.dot(A, Q[t]), A.T)

    return a, R, f, Q, m, C


def multi_filter(Y, F, G, V, W, m0=None, C0=None):
    '''
    Peforms filtering for univariate observational specifications
    considering multiple 'samples' from the observational variable
    at each time.

    Args:
        Y: A list with the vector of observations for each time.
        F: The vector that specifies the univariate behavior.
        G: The usual evolutional matrix.
        V: The observational variance for an univariate observation.
        W: The usual evolutional error covariance matrix.
        m0: The usual prior mean for the states. Defaults to zeros.
        C0: The usual prior covariance for the states. Defaults to a
            diagonal matrix with entries equal to 10**6.

    Returns:
        Four matrices - the prior means and covariances, a and R, and
        the online means and covariances, m and C.
    '''

    p = F.size
    T = len(Y)
    Gt = G.T

    if m0 is None:
        m0 = np.zeros(p)
    if C0 is None:
        C0 = np.diag(np.ones(p)) * 10**6

    a = np.empty((T, p))
    R = np.empty((T, p, p))
    m = np.empty((T, p))
    C = np.empty((T, p, p))

    n = Y[0].size
    FF = np.tile(F, (n, 1))
    a[0] = np.dot(G, m0)
    R[0] = np.dot(np.dot(G, C0), Gt) + W
    f = np.dot(FF, a[0])
    Q = np.dot(np.dot(FF, R[0]), FF.T) + V * np.eye(n)
    e = Y[0] - f
    Qinv = np.linalg.inv(Q)
    A = np.dot(np.dot(R[0], FF.T), Qinv)
    m[0] = a[0] + np.dot(A, e)
    C[0] = R[0] - np.dot(np.dot(A, Q), A.T)

    for t in range(1, T):
        n = Y[t].size
        FF = np.tile(F, (n, 1))
        a[t] = np.dot(G, m[t-1])
        R[t] = np.dot(np.dot(G, C[t-1]), Gt) + W
        f = np.dot(FF, a[t])
        Q = np.dot(np.dot(FF, R[t]), FF.T) + V * np.eye(n)
        e = Y[t] - f
        Qinv = np.linalg.inv(Q)
        A = np.dot(np.dot(R[t], FF.T), Qinv)
        m[t] = a[t] + np.dot(A, e)
        C[t] = R[t] - np.dot(np.dot(A, Q), A.T)

    return a, R, m, C


def filter_full(Y, F, G, df=0.8, m0=None, C0=None, l0=None, s0=None):
    '''
    Peforms Kalman filtering with online inference for observational variance
    and discount factors for a Dynamic Linear Model.

    Args:
        Y: The matrix of observations, with an observation in each row. Only
           supports univariate observations for the time being.
        F: The observational matrix.
        G: The evolutional matrix.
        df: The discount factor. Defaults to 0.9.
        m0: The prior mean for the states. Defaults to zeros.
        C0: The prior covariance for the states. Defaults to a diagonal
            matrix with entries equal to 10**6.
        l0: The prior 'number of entries' for observational variance.
        s0: The prior expected value for observational variance.

    Returns:
        Eight arrays - the prior means and covariances, a and R, the one
        step ahead forecast means and covariances, f and R, the
        online means and covariances, m and C, and the online 'number of
        observations' and estimates for observational variance, l and s.
    '''

    T = Y.shape[0]
    n, p = F.shape
    Gt = G.T
    Ft = F.T

    if Y.shape[1] != 1:
        raise ValueError("Only the univariate case is supported for now")
    if Y.shape[1] != n:
        raise ValueError("F dimension mismatch")
    if G.shape[0] != p or G.shape[1] != p:
        raise ValueError("G dimension mismatch")

    if m0 is None:
        m0 = np.zeros(p)
    if C0 is None:
        C0 = np.diag(np.ones(p)) * 10**6
    if l0 is None:
        l0 = 1
    if s0 is None:
        s0 = np.abs(Y[0,0])

    a = np.empty((T, p))
    R = np.empty((T, p, p))
    f = np.empty(T)
    Q = np.empty(T)
    m = np.empty((T, p))
    C = np.empty((T, p, p))
    l = np.empty(T)
    s = np.empty(T)
    W = np.empty(T)

    a[0] = np.dot(G, m0)
    P = np.dot(np.dot(G, C0), Gt)
    W[0] = P * (1 - df) / df
    R[0] = P + W[0]
    f[0] = np.dot(F, a[0])
    Q[0] = np.dot(np.dot(F, R[0]), Ft) + s0
    e = Y[0] - f[0]
    A = np.dot(R[0], Ft) / Q[0]
    m[0] = a[0] + np.dot(A, e)
    l[0] = l0 + 1
    s[0] = s0 + s0 / l[0] * (e**2 / Q[0] - 1)
    C[0] = (R[0] - Q[0] * np.dot(A, A.T)) * s[0] / s0

    for t in range(1, T):
        a[t] = np.dot(G, m[t-1])
        P = np.dot(np.dot(G, C[t-1]), Gt)
        W[t] = P * (1 - df) / df
        R[t] = P + W[t]
        f[t] = np.dot(F, a[t])
        Q[t] = np.dot(np.dot(F, R[t]), Ft) + s[t-1]
        e = Y[t] - f[t]
        A = np.dot(R[t], Ft) / Q[t]
        m[t] = a[t] + np.dot(A, e)
        l[t] = l[t-1] + 1
        s[t] = s[t-1] + s[t-1] / l[t] * (e**2 / Q[t] - 1)
        C[t] = (R[t] - Q[t] * np.dot(A, A.T)) * s[t] / s[t-1]

    return a, R, f, Q, m, C, l, s, W


def filter_partial(Y, F, G, W, m0=None, C0=None, l0=None, s0=None):
    '''
    Peforms Kalman filtering with online inference for observational variance
    for a Dynamic Linear Model.

    Args:
        Y: The matrix of observations, with an observation in each row. Only
           supports univariate observations for the time being.
        F: The observational matrix.
        G: The evolutional matrix.
        W: The evolutional covariance matrix.
        m0: The prior mean for the states. Defaults to zeros.
        C0: The prior covariance for the states. Defaults to a diagonal
            matrix with entries equal to 10**6.
        l0: The prior 'number of entries' for observational variance.
        s0: The prior expected value for observational variance.

    Returns:
        Eight arrays - the prior means and covariances, a and R, the one
        step ahead forecast means and covariances, f and R, the
        online means and covariances, m and C, and the online 'number of
        observations' and estimates for observational variance, l and s.
    '''

    T = Y.shape[0]
    n, p = F.shape
    Gt = G.T
    Ft = F.T

    if Y.shape[1] != 1:
        raise ValueError("Only the univariate case is supported for now")
    if Y.shape[1] != n:
        raise ValueError("F dimension mismatch")
    if G.shape[0] != p or G.shape[1] != p:
        raise ValueError("G dimension mismatch")
    if W.shape[0] != p or W.shape[1] != p:
        raise ValueError("W dimension mismatch")

    if m0 is None:
        m0 = np.zeros(p)
    if C0 is None:
        C0 = np.eye(p) * 10**6
    if l0 is None:
        l0 = 1
    if s0 is None:
        s0 = np.abs(Y[0,0])

    a = np.empty((T, p))
    R = np.empty((T, p, p))
    f = np.empty(T)
    Q = np.empty(T)
    m = np.empty((T, p))
    C = np.empty((T, p, p))
    l = np.empty(T)
    s = np.empty(T)

    a[0] = np.dot(G, m0)
    R[0] = np.dot(np.dot(G, C0), Gt) + W
    f[0] = np.dot(F, a[0])
    Q[0] = np.dot(np.dot(F, R[0]), Ft) + s0
    e = Y[0] - f[0]
    A = np.dot(R[0], Ft) / Q[0]
    m[0] = a[0] + np.dot(A, e)
    l[0] = l0 + 1
    s[0] = s0 + s0 / l[0] * (e**2 / Q[0] - 1)
    C[0] = (R[0] - Q[0] * np.dot(A, A.T)) * s[0] / s0

    for t in range(1, T):
        a[t] = np.dot(G, m[t-1])
        R[t] = np.dot(np.dot(G, C[t-1]), Gt) + W
        f[t] = np.dot(F, a[t])
        Q[t] = np.dot(np.dot(F, R[t]), Ft) + s[t-1]
        e = Y[t] - f[t]
        A = np.dot(R[t], Ft) / Q[t]
        m[t] = a[t] + np.dot(A, e)
        l[t] = l[t-1] + 1
        s[t] = s[t-1] + s[t-1] / l[t] * (e**2 / Q[t] - 1)
        C[t] = (R[t] - Q[t] * np.dot(A, A.T)) * s[t] / s[t-1]

    return a, R, f, Q, m, C, l, s


def smoother(G, a, R, m, C):
    '''
    Peforms basic Kalman smoothing for a Dynamic Linear Model.

    Args:
        G: The evolutional matrix.
        a: The prior mean matrix returned by the filtering step.
        R: The prior covariance matrices returned by the filtering step.
        m: The online mean matrix returned by the filtering step.
        C: The online covariance matrices returned by the filtering step.

    Returns:
        Two matrices - the posterior means and covariances, s and S.
    '''
    T, p = m.shape
    Gt = G.T

    s = np.empty((T, p))
    S = np.empty((T, p, p))

    s[T-1] = m[T-1]
    S[T-1] = C[T-1]

    for t in range(T-2, -1, -1):
        Rinv = np.linalg.inv(R[t+1])
        B = np.dot(np.dot(C[t], Gt), Rinv)
        s[t] = m[t] + np.dot(B, s[t+1] - a[t+1])
        S[t] = C[t] - np.dot(np.dot(B, R[t+1] - S[t+1]), B.T)

    return s, S


def rw_mle(y, numit=50):
    '''
    Obtains maximum likelihood estimates for a Random Walk DLM.

    Args:
        y: The vector of observations.

    Returns:
        theta: The estimates for the states.
        V: The estimate for the observational variance.
        W: The estimate for the evolutional variance.
    '''

    y = np.array([y]).T
    F = np.array([[1]])
    G = np.array([[1]])

    # Initialize values using a filter_full
    a, R, _, _, m, C, _, s, W = filter_full(y, F, G, 0.7)
    pm, pC = smoother(G, a, R, m, C)
    theta = pm[:,0]
    V = np.array([[s[-1]]])
    W = np.array([[W.mean()]])

    # Iterate on maximums
    for i in range(numit):
        a, R, _, _, m, C, l, s = filter_partial(y, F, G, W)
        pm, _ = smoother(G, a, R, m, C)
        theta = pm[:,0]  # Get first (and only) dimension as vector
        V[0, 0] = (s[-1] * l[-1] - s[0]) / (l[-1] + l[0])

        # The evolutional variance estimator comes from the difference between
        # the states and its lagged values
        W[0,0] = np.mean((theta[:-1] - theta[1:])**2)

        # TODO: Check difference between old and new estimates and stop early
        #       if values stopped changing?

    return theta, V, W
