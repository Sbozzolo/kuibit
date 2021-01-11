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

import os
import re
import unittest

import numpy as np

from kuibit import cactus_ascii_utils as cau
from kuibit import cactus_scalars as cs
from kuibit import simdir as sd
from kuibit import timeseries as ts


class TestCactusScalar(unittest.TestCase):
    def test_OneScalar(self):

        # Filename not recogonized
        with self.assertRaises(RuntimeError):
            cs.OneScalar("123.h5")

        # Reduction not recogonized
        with self.assertRaises(RuntimeError):
            cs.OneScalar("hydrobase-press.bubu.asc")

        # maximum, vector, one file per variable
        path = "tests/tov/output-0000/static_tov/vel[0].maximum.asc"
        asc = cs.OneScalar(path)

        self.assertFalse(asc._is_one_file_per_group)
        self.assertFalse(asc._was_header_scanned)
        self.assertEqual(asc.reduction_type, "maximum")
        self.assertDictEqual(asc._vars, {"vel[0]": None})

        # no reduction, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/carpet-timing..asc"
        asc_carp = cs.OneScalar(path)

        self.assertTrue(asc_carp._is_one_file_per_group)
        self.assertTrue(asc_carp._was_header_scanned)
        self.assertIn("current_physical_time_per_hour", asc_carp._vars)
        self.assertEqual(asc_carp._vars["current_physical_time_per_hour"], 13)
        self.assertIn("time_total", asc_carp._vars)
        self.assertEqual(asc_carp._vars["time_total"], 14)
        self.assertIs(asc_carp.reduction_type, "scalar")

        # Compressed, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/hydrobase-eps.minimum.asc.gz"
        asc_gz = cs.OneScalar(path)

        self.assertTrue(asc_gz._is_one_file_per_group)
        self.assertTrue(asc_gz._was_header_scanned)
        self.assertEqual(asc_gz.reduction_type, "minimum")
        self.assertEqual(asc_gz._compression_method, "gz")
        self.assertDictEqual(asc_gz._vars, {"eps": 2})

        # Compressed, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/hydrobase-eps.minimum.asc.bz2"
        asc_bz = cs.OneScalar(path)
        self.assertEqual(asc_bz._compression_method, "bz2")
        self.assertDictEqual(asc_bz._vars, {"eps": 2})

    def test_OneScalar_magic_methods(self):

        path = "tests/tov/output-0000/static_tov/vel[0].maximum.asc"
        asc = cs.OneScalar(path)

        self.assertIn("vel[0]", asc)

        self.assertCountEqual(asc.keys(), ["vel[0]"])

    def test__scan_strings_for_columns(self):

        rx_columns = re.compile(r"^(\d+):(\w+(\[\d+\])?)$")
        rx_data_columns = re.compile(r"^# data columns: (.+)$")

        # Not matching strings
        strings = ["bubu"]
        with self.assertRaises(RuntimeError):
            cau._scan_strings_for_columns(strings, rx_columns)

        # Not matching columns
        strings = ["# data columns: bubu:press"]
        with self.assertRaises(RuntimeError):
            cau._scan_strings_for_columns(strings, rx_data_columns)

        # Good columns
        strings = ["# data columns: 3:press"]
        self.assertDictEqual(
            cau._scan_strings_for_columns(strings, rx_data_columns),
            {"press": 2},
        )

    def test__scan_header(self):
        # This also tests the module scan_header

        # __init__ scans the header for some files, so to debug this it may be
        # useful to comment that section temporarily

        # Here we test if the errors are raised

        path = "no-time..asc"

        with open(path, "wt") as test_file:
            test_file.write("# column format: 1:data")

        with self.assertRaises(RuntimeError):
            cs.OneScalar(path)
        os.remove(path)

        path = "no-data..asc"

        with open(path, "wt") as test_file:
            test_file.write("# column format: 1:time")

        with self.assertRaises(RuntimeError):
            cs.OneScalar(path)

        os.remove(path)

    def test_load(self):

        # no reduction, scalar, one file per group
        path = "tests/tov/output-0000/static_tov/carpet-timing..asc"
        asc_carp = cs.OneScalar(path)
        t, y = np.loadtxt(path, ndmin=2, unpack=True, usecols=(8, 13))

        self.assertEqual(
            asc_carp.load("current_physical_time_per_hour"),
            ts.TimeSeries(t, y),
        )

        # Test __getitem__
        self.assertEqual(
            asc_carp["current_physical_time_per_hour"], ts.TimeSeries(t, y)
        )

        # Value not existing
        with self.assertRaises(ValueError):
            asc_carp.load("bubu")

        # Test scanning header
        path = "tests/tov/output-0000/static_tov/vel[0].maximum.asc"
        asc = cs.OneScalar(path)
        vel = asc.load("vel[0]")

    def test_AllScalars(self):

        sim = sd.SimDir("tests/tov")

        reader = cs.AllScalars(sim.allfiles, "average")

        # Let's check that all the files are properly indexed
        vars_tov = [
            "H",
            "kxx",
            "kxy",
            "kxz",
            "kyy",
            "kyz",
            "kzz",
            "press",
            "alp",
            "gxx",
            "gxy",
            "gxz",
            "gyy",
            "gyz",
            "gzz",
            "M1",
            "M2",
            "M3",
            "eps",
            "rho",
            "vel[0]",
            "vel[1]",
            "vel[2]",
        ]

        self.assertCountEqual(reader._vars, vars_tov)

        self.assertCountEqual(reader.keys(), vars_tov)

        self.assertTrue(
            reader.__str__().startswith("Available average timeseries:\n[")
        )

    def test_AllScalars_magic_methods(self):

        reader = cs.AllScalars(sd.SimDir("tests/tov").allfiles, "average")
        self.assertIn("rho", reader)

        path1 = "tests/tov/output-0000/static_tov/hydrobase-rho.average.asc"
        path2 = "tests/tov/output-0001/static_tov/hydrobase-rho.average.asc"
        t1, y1 = np.loadtxt(path1, ndmin=2, unpack=True, usecols=(1, 2))
        t2, y2 = np.loadtxt(path2, ndmin=2, unpack=True, usecols=(1, 2))

        rho = ts.TimeSeries(np.append(t1, t2), np.append(y1, y2))

        self.assertEqual(rho, reader["rho"])
        self.assertEqual(rho, reader.get("rho"))

        self.assertEqual(1, reader.get("bubu", default=1))

    def test_ScalarsDir(self):

        # Not a SimDir
        with self.assertRaises(TypeError):
            cs.ScalarsDir(0)

        scaldir = cs.ScalarsDir(sd.SimDir("tests/tov"))

        # Check that the getter (and []) work
        self.assertEqual(scaldir["average"].reduction_type, "average")
        self.assertEqual(scaldir.get("infnorm").reduction_type, "infnorm")
        self.assertIsNone(scaldir.get("bubu", default=None))

        self.assertIs(scaldir["maximum"], scaldir["max"])
        self.assertIs(scaldir["minimum"], scaldir["min"])

        # Check string representation
        # (this is a very weak check...)
        self.assertIn("io_count", scaldir.__str__())
