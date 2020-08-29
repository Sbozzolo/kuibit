#!/usr/bin/env python3

import unittest
import numpy as np
import os
from postcactus import timeseries as ts
from postcactus import simdir as sd
from postcactus import cactus_multipoles as mp


class TestCactusMultipoles(unittest.TestCase):

    def test_MultipoleDet(self):

        # Prepare fake multipoles
        t1 = np.linspace(0, 1, 100)
        t2 = np.linspace(2, 3, 100)
        y1 = np.sin(t1)
        y2 = np.sin(t2)
        ts1 = ts.TimeSeries(t1, y1)
        ts2 = ts.TimeSeries(t2, y2)

        ts_comb = ts.combine_ts([ts1, ts2])

        data = [(2, 2, ts1), (2, 2, ts2)]
        data2 = [(2, 2, ts1), (2, -2, ts2)]

        mult1 = mp.MultipoleDet(100, data)
        mult2 = mp.MultipoleDet(100, data2)

        self.assertEqual(mult1.dist, 100)
        self.assertEqual(mult1.radius, 100)

        # test __call__
        self.assertEqual(mult1(2, 2), ts_comb)
        self.assertEqual(mult2(2, 2), ts1)
        self.assertEqual(mult2(2, -2), ts2)

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
        self.assertEqual(mult2[(2, 2)], ts1)

        # test_iter
        expected = [(2, 2, ts1), (2, -2, ts2)]
        for data, exp in zip(mult2, expected):
            self.assertCountEqual(data, exp)

        # test __len__
        self.assertEqual(len(mult1), 1)
        self.assertEqual(len(mult2), 2)

        # test keys()
        self.assertCountEqual(mult1.keys(), [(2,2)])
