#!/usr/bin/env python3

import unittest
import numpy as np
import h5py
from postcactus import timeseries as ts
from postcactus import simdir as sd
from postcactus import cactus_multipoles as mp
import time


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
        data3 = [(2, 2, self.ts1), (1, 1, self.ts2)]

        mult1 = mp.MultipoleDet(100, data)
        mult2 = mp.MultipoleDet(100, data2)
        mult3 = mp.MultipoleDet(100, data3, l_min=2)

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
        self.assertCountEqual(mult3.missing_lm, {(2, -2),
                                                 (2, -1),
                                                 (2, 0),
                                                 (2, 1)})

        # test _warn_missing
        with self.assertWarns(Warning):
            mult3._warn_missing("Energy")


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

        # test __str__()
        self.assertIn("(2, 2)", mult1.__str__())
        self.assertIn("missing", mult3.__str__())

    def test_MultipoleAllDet(self):

        data = [(2, 2, 100, self.ts1),
                (2, -2, 150, self.ts2)]

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

    def test_MultipolesDir(self):

        sim = sd.SimDir("tests/tov")
        cacdir = mp.MultipolesDir(sim)

        # multipoles from textfile
        with self.assertRaises(RuntimeError):
            cacdir._multipole_from_textfile("tests/tov/output-0000/static_tov/carpet-timing..asc")

        path = "tests/tov/output-0000/static_tov/mp_Phi2_l2_m-1_r110.69.asc"
        path_h5 = "tests/tov/output-0000/static_tov/mp_harmonic.h5"
        t, real, imag = np.loadtxt(path).T

        with h5py.File(path_h5, 'r') as data:
            # Loop over the groups in the hdf5
            a = data["l2_m2_r8.00"][()].T

        mpts = ts.TimeSeries(t, real + 1j * imag)
        ts_h5 = ts.TimeSeries(a[0], a[1] + 1j*a[2])

        self.assertEqual(mpts, cacdir._multipole_from_textfile(path))
        self.assertEqual(ts_h5,
                         cacdir._multipoles_from_h5files([path_h5])[8.00](2,2))

        mpfiles = [(2, 2, 100, path)]

        # Check one specific case
        self.assertEqual(mpts,
                         cacdir._multipoles_from_textfiles(mpfiles)[100](2, 2))

        self.assertEqual(cacdir['phi2'][110.69](2, -1), mpts)
        self.assertEqual(cacdir['harmonic'][8.00](2, 2), ts_h5)

        # test get
        self.assertIs(cacdir.get("bubu"), None)
        self.assertEqual(cacdir.get('harmonic')[8.00](2, 2), ts_h5)

        # test __getitem__
        with self.assertRaises(KeyError):
            cacdir['bubu']

        # test __contains__
        self.assertIn('phi2', cacdir)

        # test keys()
        self.assertCountEqual(cacdir.keys(), ['harmonic', 'phi2', 'psi4'])

        # test __str__()
        self.assertIn("harmonic", cacdir.__str__())
