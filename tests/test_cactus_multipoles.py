#!/usr/bin/env python3

import unittest
import numpy as np
import os
from postcactus import timeseries as ts
from postcactus import simdir as sd
from postcactus import cactus_multipoles as mp


class TestCactusMultipoles(unittest.TestCase):

    def setUp(self):
        # Prepare fake multipoles
        self.t1 = np.linspace(0, 1, 100)
        self.t2 = np.linspace(2, 3, 100)
        self.y1 = np.sin(self.t1)
        self.y2 = np.sin(self.t2)
        self.ts1 = ts.TimeSeries(self.t1, self.y1)
        self.ts2 = ts.TimeSeries(self.t2, self.y2)
        
    def test_MultipoleDet(self):

        ts_comb = ts.combine_ts([self.ts1, self.ts2])

        data = [(2, 2, self.ts1), (2, 2, self.ts2)]
        data2 = [(2, 2, self.ts1), (2, -2, self.ts2)]

        mult1 = mp.MultipoleDet(100, data)
        mult2 = mp.MultipoleDet(100, data2)

        self.assertEqual(mult1.dist, 100)
        self.assertEqual(mult1.radius, 100)

        # test __call__
        self.assertEqual(mult1(2, 2), ts_comb)
        self.assertEqual(mult2(2, 2), self.ts1)
        self.assertEqual(mult2(2, -2), self.ts2)

        # test available_
        self.assertCountEqual(mult1.available_l, {2})
        self.assertCountEqual(mult2.available_l, {2})
        self.assertCountEqual(mult1.available_m, {2})
        self.assertCountEqual(mult2.available_m, {2, -2})
        self.assertCountEqual(mult1.available_lm, {(2, 2)})
        self.assertCountEqual(mult2.available_lm, {(2, 2), (2, -2)})

        # test contains
        self.assertIn((2, 2), mult1)
        self.assertIn((2, -2), mult2)
        self.assertEqual(mult2[(2, 2)], self.ts1)

        # test_iter
        expected = [(2, 2, self.ts1), (2, -2, self.ts2)]
        for data, exp in zip(mult2, expected):
            self.assertCountEqual(data, exp)

        # test __len__
        self.assertEqual(len(mult1), 1)
        self.assertEqual(len(mult2), 2)

        # test keys()
        self.assertCountEqual(mult1.keys(), [(2, 2)])

        # test __eq__()
        self.assertNotEqual(mult1, mult2)
        self.assertNotEqual(mult1, 1)
        self.assertEqual(mult1, mult1)

    def test_MultipoleAllDet(self):

        data = [(2, 2, 100, self.ts1),
                (2, -2, 150, self.ts2)]

        radii = [100, 150]

        alldets = mp.MultipoleAllDets(data)

        self.assertEqual(alldets.radii, radii)
        self.assertSetEqual(alldets.available_lm, {(2, 2), (2, -2)})

        # test __contains__
        self.assertIn(100, alldets)

        # test __getitem__
        data_single = [(2, 2, self.ts1)]
        mult_single = mp.MultipoleDet(100, data_single)
        self.assertEqual(alldets[100], mult_single)

        # test __eq__
        self.assertEqual(alldets, alldets)
        self.assertNotEqual(alldets, 1)

        data2 = [(2, 2, 100, self.ts1),
                 (2, -2, 180, self.ts2)]

        alldets2 = mp.MultipoleAllDets(data2)

        self.assertNotEqual(alldets, alldets2)


        # test __iter__
        for det, r in zip(alldets, radii):
            self.assertEqual(det.radius, r)

        # test __len__
        self.assertEqual(len(alldets), 2)

        # keys()
        self.assertCountEqual(alldets.keys(), radii)
