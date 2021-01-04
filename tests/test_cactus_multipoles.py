#!/usr/bin/env python3

# Copyright (C) 2020-2021 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

import unittest

import h5py
import numpy as np

from postcactus import cactus_multipoles as mp
from postcactus import simdir as sd
from postcactus import timeseries as ts


class TestCactusMultipoles(unittest.TestCase):
    def setUp(self):

        # Prepare fake multipoles
        self.t1 = np.linspace(0, 1, 100)
        self.t2 = np.linspace(2, 3, 100)
        self.y1 = np.sin(self.t1)
        self.y2 = np.sin(self.t2)
        self.ts1 = ts.TimeSeries(self.t1, self.y1)
        self.ts2 = ts.TimeSeries(self.t2, self.y2)

    def test_MultipoleOneDet(self):

        ts_comb = ts.combine_ts([self.ts1, self.ts2])

        data = [(2, 2, self.ts1), (2, 2, self.ts2)]
        data2 = [(2, 2, self.ts1), (2, -2, self.ts2)]
        data3 = [(2, 2, self.ts1), (1, 1, self.ts2)]

        # Combinging ts
        mult1 = mp.MultipoleOneDet(100, data)
        # Different multipoles
        mult2 = mp.MultipoleOneDet(100, data2)
        # l_min != 0
        mult3 = mp.MultipoleOneDet(100, data3, l_min=2)

        self.assertEqual(mult1.dist, 100)
        self.assertEqual(mult1.radius, 100)
        self.assertEqual(mult1.l_min, 0)
        self.assertEqual(mult3.l_min, 2)

        # test __call__
        self.assertEqual(mult1(2, 2), ts_comb)
        self.assertEqual(mult2(2, 2), self.ts1)
        self.assertEqual(mult2(2, -2), self.ts2)

        # test copy()
        self.assertEqual(mult1.copy(), mult1)
        self.assertIsNot(mult1.copy(), mult1)

        # test available_
        self.assertCountEqual(mult1.available_l, {2})
        self.assertCountEqual(mult2.available_l, {2})
        self.assertCountEqual(mult1.available_m, {2})
        self.assertCountEqual(mult2.available_m, {2, -2})
        self.assertCountEqual(mult1.available_lm, {(2, 2)})
        self.assertCountEqual(mult2.available_lm, {(2, 2), (2, -2)})
        self.assertCountEqual(mult3.available_l, {2})
        self.assertCountEqual(mult3.available_m, {2})
        self.assertCountEqual(
            mult3.missing_lm, {(2, -2), (2, -1), (2, 0), (2, 1)}
        )

        # test contains
        self.assertIn((2, 2), mult1)
        self.assertIn((2, -2), mult2)
        self.assertEqual(mult2[(2, 2)], self.ts1)

        # test_iter
        # Notice the order. It is increasing in (l, m)
        expected = [(2, -2, self.ts2), (2, 2, self.ts1)]
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

        # test __str__()
        self.assertIn("(2, 2)", mult1.__str__())
        self.assertIn("missing", mult3.__str__())

    def test_total_function_on_available_lm(self):

        # The two series must have the same times
        ts3 = ts.TimeSeries(self.t1, self.y2)
        data = [(2, 2, self.ts1), (1, -1, ts3)]
        mult = mp.MultipoleOneDet(100, data)

        # Test larger and smaller l
        with self.assertRaises(ValueError):
            mult.total_function_on_available_lm(lambda x: x, l_max=100)
        with self.assertRaises(ValueError):
            mult.total_function_on_available_lm(lambda x: x, l_max=0)

        # First, let's try with identity as function
        # The output should be just ts1 + ts2
        def identity(x, *args):
            return x

        self.assertEqual(
            mult.total_function_on_available_lm(identity),
            self.ts1 + ts3,
        )

        # Next, we use the l, m, r, information
        def func1(x, mult_l, mult_m, mult_r):
            return mult_l * mult_m * mult_r * x

        self.assertEqual(
            mult.total_function_on_available_lm(func1),
            2 * 2 * 100 * self.ts1 + 1 * (-1) * 100 * ts3,
        )

        # Next, we use args and kwargs
        def func2(x, mult_l, mult_m, mult_r, add, add2=0):
            return mult_l * mult_m * mult_r * x + add + add2

        # This will just add (add + add2) (2 + 3)
        self.assertEqual(
            mult.total_function_on_available_lm(func2, 2, add2=3),
            (2 * 2 * 100 * self.ts1 + 2 + 3 + 1 * (-1) * 100 * ts3 + 2 + 3),
        )

        # Finally, test l_max
        self.assertEqual(
            mult.total_function_on_available_lm(identity, l_max=1),
            ts3,
        )

    def test_MultipoleAllDets(self):

        data = [(2, 2, 100, self.ts1), (2, -2, 150, self.ts2)]

        radii = [100, 150]

        alldets = mp.MultipoleAllDets(data)

        self.assertEqual(alldets.radii, radii)
        self.assertSetEqual(alldets.available_lm, {(2, 2), (2, -2)})

        # test __contains__
        self.assertIn(100, alldets)

        # test copy()
        self.assertEqual(alldets.copy(), alldets)
        self.assertIsNot(alldets.copy(), alldets)

        # test __getitem__
        data_single = [(2, 2, self.ts1)]
        mult_single = mp.MultipoleOneDet(100, data_single)
        self.assertEqual(alldets[100], mult_single)

        # test __eq__
        self.assertEqual(alldets, alldets)
        self.assertNotEqual(alldets, 1)

        data2 = [(2, 2, 100, self.ts1), (2, -2, 180, self.ts2)]

        alldets2 = mp.MultipoleAllDets(data2)

        self.assertNotEqual(alldets, alldets2)

        # test __iter__
        for det, r in zip(alldets, radii):
            self.assertEqual(det.radius, r)

        # test __len__
        self.assertEqual(len(alldets), 2)

        # keys()
        self.assertCountEqual(alldets.keys(), radii)

    def test_has_detector(self):

        data = [(2, 2, 100, self.ts1), (2, -2, 150, self.ts2)]

        alldets = mp.MultipoleAllDets(data)

        self.assertFalse(alldets.has_detector(2, 2, 0))
        self.assertFalse(alldets.has_detector(2, 3, 100))
        self.assertTrue(alldets.has_detector(2, 2, 100))

    def test_MultipolesDir(self):

        sim = sd.SimDir("tests/tov")
        cacdir = mp.MultipolesDir(sim)

        # multipoles from textfile
        with self.assertRaises(RuntimeError):
            cacdir._multipole_from_textfile(
                "tests/tov/output-0000/static_tov/carpet-timing..asc"
            )

        path = "tests/tov/output-0000/static_tov/mp_Phi2_l2_m-1_r110.69.asc"
        path_h5 = "tests/tov/output-0000/static_tov/mp_harmonic.h5"
        t, real, imag = np.loadtxt(path).T

        with h5py.File(path_h5, "r") as data:
            # Loop over the groups in the hdf5
            a = data["l2_m2_r8.00"][()].T

        mpts = ts.TimeSeries(t, real + 1j * imag)
        ts_h5 = ts.TimeSeries(a[0], a[1] + 1j * a[2])

        self.assertEqual(mpts, cacdir._multipole_from_textfile(path))
        self.assertEqual(
            ts_h5,
            cacdir._multipoles_from_h5files([path_h5])[8.00](2, 2),
        )

        mpfiles = [(2, 2, 100, path)]

        # Check one specific case
        self.assertEqual(
            mpts,
            cacdir._multipoles_from_textfiles(mpfiles)[100](2, 2),
        )

        self.assertEqual(cacdir["phi2"][110.69](2, -1), mpts)
        self.assertEqual(cacdir["harmonic"][8.00](2, 2), ts_h5)

        # test get
        self.assertIs(cacdir.get("bubu"), None)
        self.assertEqual(cacdir.get("harmonic")[8.00](2, 2), ts_h5)

        # test __getitem__
        with self.assertRaises(KeyError):
            cacdir["bubu"]

        # test __contains__
        self.assertIn("phi2", cacdir)

        # test keys()
        self.assertCountEqual(cacdir.keys(), ["harmonic", "phi2", "psi4"])

        # test __str__()
        self.assertIn("harmonic", cacdir.__str__())
