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

import scipy.stats as sps
import scipy.optimize as opt


def simulate(T, F, G, V, W, theta_init=None, n=None):
    '''
    Simulates observations for a Dynamic Linear Model.

    Args:
        T: The size of the time window.
        F: The observational matrix.
        G: The evolutional matrix.
        V: The observational error covariance matrix.
        W: The evolutional error covariance matrix.
        theta_init: The value of "theta_0". Defaults to zeros.
        n: Number of replicates for observations. Defaults to no replicates.

    Returns:
        Y: The observations.
        theta: The states.
    '''

    m = F.shape[0]
    p = F.shape[1]

    if theta_init is None:
        theta_init = np.zeros(p)

    theta = np.empty((T, p))
    Y = np.empty((T, m)) if n is None else np.empty((T, n*m))

    if n is None:
        theta[0] = sps.multivariate_normal.rvs(np.dot(G, theta_init), W)
        Y[0] = sps.multivariate_normal.rvs(np.dot(F, theta[0]), V)
        for t in range(1, T):
            theta[t] = sps.multivariate_normal.rvs(np.dot(G, theta[t-1]), W)
            Y[t] = sps.multivariate_normal.rvs(np.dot(F, theta[t]), V)
    else:
        theta[0] = sps.multivariate_normal.rvs(np.dot(G, theta_init), W)
        Y[0] = np.hstack([sps.multivariate_normal.rvs(np.dot(F, theta[0]), V) for i in range(n)])
        for t in range(1, T):
            theta[t] = sps.multivariate_normal.rvs(np.dot(G, theta[t-1]), W)
            Y[t] = np.hstack([sps.multivariate_normal.rvs(np.dot(F, theta[t]), V) for i in range(n)])

    return Y, theta


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


def filter_df(Y, F, G, V, df=0.9, m0=None, C0=None):
    '''
    Peforms the basic Kalman filter for a Dynamic Linear Model with discount
    factor modelling for the evolutional variance.

    Args:
        Y: The matrix of observations, with an observation in each row.
        F: The observational matrix.
        G: The evolutional matrix.
        V: The observational error covariance matrix.
        df: The discount factor. Defaults to 0.9.
        m0: The prior mean for the states. Defaults to zeros.
        C0: The prior covariance for the states. Defaults to a diagonal
            matrix with entries equal to 10**6.

    Returns:
        a: Prior means.
        R: Prior covariances.
        f: One-step ahead forecast means.
        Q: One-step ahead forecast covariances.
        m: Online means.
        C: Online covariances.
        W: Imposed values for W.
    '''

    T = Y.shape[0]
    n, p = F.shape
    Gt = G.T
    Ft = F.T

    if Y.shape[1] != n or V.shape[0] != n or V.shape[1] != n:
        raise ValueError("Observational dimension mismatch")
    if G.shape[0] != p or G.shape[1] != p:
        raise ValueError("G dimension mismatch")

    if m0 is None:
        m0 = np.zeros(p)
    elif type(m0) in [float, int]:
        m0 = np.ones(p) * m0

    if C0 is None:
        C0 = np.eye(p) * 10**6
    elif type(C0) in [float, int]:
        C0 = np.eye(p) * C0

    a = np.empty((T, p))
    R = np.empty((T, p, p))
    f = np.empty((T, n))
    Q = np.empty((T, n, n))
    m = np.empty((T, p))
    C = np.empty((T, p, p))
    W = np.empty((T, p, p))

    a[0] = np.dot(G, m0)
    P = np.dot(np.dot(G, C0), Gt)
    W[0] = P * (1 - df) / df
    R[0] = np.dot(np.dot(G, C0), Gt) + W[0]
    f[0] = np.dot(F, a[0])
    Q[0] = np.dot(np.dot(F, R[0]), Ft) + V
    e = Y[0] - f[0]
    Qinv = np.linalg.inv(Q[0])
    A = np.dot(np.dot(R[0], Ft), Qinv)
    m[0] = a[0] + np.dot(A, e)
    C[0] = R[0] - np.dot(np.dot(A, Q[0]), A.T)

    for t in range(1, T):
        a[t] = np.dot(G, m[t-1])
        P = np.dot(np.dot(G, C[t-1]), Gt)
        W[t] = P * (1 - df) / df
        R[t] = np.dot(np.dot(G, C[t-1]), Gt) + W[t]
        f[t] = np.dot(F, a[t])
        Q[t] = np.dot(np.dot(F, R[t]), Ft) + V
        e = Y[t] - f[t]
        Qinv = np.linalg.inv(Q[t])
        A = np.dot(np.dot(R[t], Ft), Qinv)
        m[t] = a[t] + np.dot(A, e)
        C[t] = R[t] - np.dot(np.dot(A, Q[t]), A.T)

    return a, R, f, Q, m, C, W


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
        s0 = np.abs(Y[0, 0])

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
        s0 = np.abs(Y[0, 0])

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


def likelihood(y, theta, F, G, V, W):
    '''
    Log-likelihood function for a general DLM.

    Args:
        y: The vector of observations.
        theta: The vector of states.
        F: The observational matrix.
        G: The evolutional matrix.
        V: The observational covariance matrix.
        W: The evolutional covariance matrice (or matrices).

    Returns:
        The log-likelihood value.
    '''

    T = y.shape[0]

    # A constant covariance matrix was passed, tile it T times
    if W.ndim == 2:
        W = np.tile(W, (T, 1, 1))

    logpdf = sps.multivariate_normal.logpdf(y[0], np.dot(F, theta[0]), V)

    for t in range(1, T):
        logpdf += \
            sps.multivariate_normal.logpdf(y[t], np.dot(F, theta[t]), V) + \
            sps.multivariate_normal.logpdf(theta[t], np.dot(G, theta[t-1]), W[t])

    return logpdf


def mle(y, F, G, df=0.7, m0=None, C0=None, maxit=50, numeps=1e-10,
        verbose=False):
    '''
    Obtains maximum likelihood estimates for a general DLM assuming
    discount factor for the latent state evolution and using coordinate
    descent with analytical steps.

    Note that the observational matrix is assumed to be constant.

    Args:
        y: The vector of observations.
        F: The observational matrix.
        G: The evolutional matrix.
        df: Discount factor.
        m0: Passed onto filter_df. Defaults to None.
        C0: Passed onto filter_df. Defaults to None.
        maxit: Maximum number of iterations. Defaults to 100.
        numeps: Small numerical value for convergence purposes. Defaults to 10**-10.
        verbose: Print out information about execution.

    Returns:
        theta: The estimates for the states.
        V: The estimate for the observational variance.
        W: The fixed values of W.
        converged: Boolean stating if the algorithm converged.
    '''

    T = y.shape[0]

    # Initialize values
    V = np.eye(y.shape[1])
    theta = np.ones((y.shape[0], G.shape[0]))

    # Iterate on maximums
    for it in range(maxit):
        old_theta = theta

        # Maximum for states is the mean for the normal
        a, R, _, _, m, C, W = filter_df(y, F, G, V, df, m0, C0)
        theta, _ = smoother(G, a, R, m, C)

        # The observational variance estimator comes from the inverse gamma
        # distribution. We have that V | theta, y ~ IG(n, np.sum((y - theta)**2))
        # and the mode is beta / (alpha + 1)
        V = np.zeros(V.shape)
        for t in range(T):
            V += np.diag((y[t] - np.dot(F, theta[t]))**2 / T)

        # Stop if convergence condition is satisfied
        if np.mean((theta - old_theta)**2) < numeps**2:
            if verbose:
                print(f'Convergence condition reached in {it} iterations.')
            break
    else:
        print(f'Convergence condition NOT reached in {maxit} iterations.')
        return theta, V, W, False

    return theta, V, W, True


def weighted_mle(y, F, G, weights, df=0.7, m0=None, C0=None, maxit=50,
                 numeps=1e-10, verbose=False):
    '''
    Obtains weighted maximum likelihood estimates for a general DLM
    assuming a discount factor for the latent state evolution and using
    coordinate descent with analytical steps and assuming that the
    observations are replicated from the same state at each time point.

    Args:
        y: The vector of observations.
        F: The observational matrix.
        G: The evolutional matrix.
        weights: The weight of each observation.
        df: Discount factor.
        m0: Passed onto filter_df. Defaults to None.
        C0: Passed onto filter_df. Defaults to None.
        maxit: Maximum number of iterations. Defaults to 100.
        numeps: Small numerical value for convergence purposes. Defaults to 10**-10.
        verbose: Print out information about execution.

    Returns:
        theta: The estimates for the states.
        V: The estimate for the observational variance.
        W: The fixed values of W.
        converged: Boolean stating if the algorithm converged.
    '''

    # Dimension of a single observation at one time point
    m = F.shape[0]

    # Number of observations
    n = y.shape[1] / m
    if int(n) != n:
        raise ValueError('Dimension of y not multiple of dimension implied by F')
    n = int(n)

    # Observation masks: associates observation index with matrix indexes
    index_mask = {i: range(i * m, (i + 1) * m) for i in range(n)}

    if verbose:
        print(f'There are {n} observations each with dimension {m}.')

    # Process weight vector
    if type(weights) in (int, float):
        weights = np.repeat(weights, n)
    elif len(weights) != n:
        raise ValueError('Incorrect length for weight vector')

    # Time dimension
    T = y.shape[0]

    # State dimension
    p = F.shape[1]

    # Initialize values
    vars = np.ones(m)
    theta = np.ones((T, p))

    # Build the tiled observational matrix
    FF = np.tile(F, (n, 1))

    # Iterate on maximums
    for it in range(maxit):
        old_theta = theta

        # Build weighted observational matrix
        weighted_vars = np.tile(vars, n) * np.repeat(1 / weights, m)
        V = np.diag(weighted_vars)

        # Maximum for states is the mean for the normal
        a, R, _, _, M, C, W = filter_df(y, FF, G, V, df, m0, C0)
        theta, _ = smoother(G, a, R, M, C)

        # Maximum for the variances
        vars = np.zeros(m)
        for i in range(n):
            mask = index_mask[i]
            for t in range(T):
                vars += weights[i] * (y[t,mask] - np.dot(F, theta[t]))**2 / T
        vars /= weights.sum()

        # Stop if convergence condition is satisfied
        if np.mean((theta - old_theta)**2) < numeps**2:
            if verbose:
                print(f'Convergence condition reached in {it} iterations.')
            break
    else:
        print(f'Convergence condition NOT reached in {maxit} iterations.')
        return theta, np.diag(vars), W, False

    return theta, np.diag(vars), W, True
