'''
Peforms tests for the dirichlet module.

Copyright (c) Victhor S. Sartório. All rights reserved.
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
'''

import unittest
import numpy as np
import dynmix as dm


class DirichletTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_dirichlet_forward_filter(self):
        y = np.array([[1, 3, 0, 6],
                      [1, 1, 2, 6],
                      [0, 2, 3, 5],
                      [1, 2, 4, 3],
                      [1, 4, 0, 5],
                      [2, 2, 3, 3],
                      [0, 2, 5, 3],
                      [4, 1, 3, 2],
                      [3, 4, 0, 3],
                      [0, 2, 5, 3],
                      [0, 3, 5, 2],
                      [1, 3, 2, 4],
                      [2, 4, 0, 4],
                      [2, 5, 2, 1],
                      [1, 4, 1, 4],
                      [1, 5, 1, 3],
                      [1, 1, 1, 7],
                      [2, 1, 2, 5],
                      [1, 1, 4, 4],
                      [0, 3, 4, 3]])
        correct = np.array([[1.09, 3.09, 0.09, 6.09],
                            [1.981, 3.781, 2.081, 11.481],
                            [1.7829, 5.4029, 4.8729, 15.3329],
                            [2.60461, 6.86261, 8.38561, 16.79961],
                            [3.344149, 10.176349, 7.547049, 20.119649],
                            [5.0097341, 11.1587141, 9.7923441, 21.1076841],
                            [4.50876069, 12.04284269, 13.81310969, 21.99691569],
                            [8.05788462, 11.83855842, 15.43179872, 21.79722412],
                            [10.25209616, 14.65470258, 13.88861885, 22.61750171],
                            [9.22688654, 15.18923232, 17.49975696, 23.35575154],
                            [8.30419789, 16.67030909, 20.74978127, 23.02017638],
                            [8.4737781, 18.00327818, 20.67480314, 24.71815875],
                            [9.62640029, 20.20295036, 18.60732283, 26.24634287],
                            [10.66376026, 23.18265533, 18.74659054, 24.62170858],
                            [10.59738423, 24.86438979, 17.87193149, 26.15953773],
                            [10.53764581, 27.37795081, 17.08473834, 26.54358395],
                            [10.48388123, 25.64015573, 16.37626451, 30.88922556],
                            [11.43549311, 24.07614016, 16.73863806, 32.800303],
                            [11.2919438, 22.66852614, 19.06477425, 33.5202727],
                            [10.16274942, 23.40167353, 21.15829683, 33.16824543]])
        result = dm.dirichlet.forward_filter(y, .9, np.array([.1, .1, .1, .1]))
        self.assertTrue(np.isclose(result, correct).all())

    def test_dirichlet_backwards_sampler(self):
        y = np.array([[1, 3, 0, 6],
                      [1, 1, 2, 6],
                      [0, 2, 3, 5],
                      [1, 2, 4, 3],
                      [1, 4, 0, 5],
                      [2, 2, 3, 3],
                      [0, 2, 5, 3],
                      [4, 1, 3, 2],
                      [3, 4, 0, 3],
                      [0, 2, 5, 3],
                      [0, 3, 5, 2],
                      [1, 3, 2, 4],
                      [2, 4, 0, 4],
                      [2, 5, 2, 1],
                      [1, 4, 1, 4],
                      [1, 5, 1, 3],
                      [1, 1, 1, 7],
                      [2, 1, 2, 5],
                      [1, 1, 4, 4],
                      [0, 3, 4, 3]])
        c = dm.dirichlet.forward_filter(y, .9, np.array([.1, .1, .1, .1]))

        num_samples = 100
        results = np.empty((num_samples, *c.shape))
        for i in range(num_samples):
            results[i] = dm.dirichlet.backwards_sampler(c, 0.9)
        mean_result = results.mean(axis = (0,1))
        self.assertTrue(0.110 <= mean_result[0] <= 0.130)
        self.assertTrue(0.250 <= mean_result[1] <= 0.270)
        self.assertTrue(0.225 <= mean_result[2] <= 0.245)
        self.assertTrue(0.370 <= mean_result[3] <= 0.390)
