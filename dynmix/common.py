'''
This module implements common utility functions for the static, independent
and dynmix modules.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import numpy as np


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



