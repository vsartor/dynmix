'''
Dynamic Membership Mixture Models:
    Allows implementation of Mixture Models for time-series data in
    which the membership of each time-series to each cluster may
    vary over time.

Copyright notice:
    Copyright (c) Victhor S. Sartório. All rights reserved. This
    Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with
    this file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

from . import gmm, dlm, dirichlet, static, independent, dynamic
