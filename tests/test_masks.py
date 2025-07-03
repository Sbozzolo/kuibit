#!/usr/bin/env python3

# Copyright (C) 2021-2025 Gabriele Bozzola
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

import numpy as np

from kuibit import grid_data as gd
from kuibit import grid_data_utils as gdu
from kuibit import masks as ma
from kuibit import timeseries as ts


class TestMasks(unittest.TestCase):
    def setUp(self):
        # Let's test with a TimeSeries, a UniformGridData, and a
        # HierarchicalGridData

        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)

        self.TS = ts.TimeSeries(x, y)

        self.grid_2d = gd.UniformGrid([10, 20], x0=[0.5, 1], dx=[1, 1])

        self.ugd = gdu.sample_function_from_uniformgrid(
            lambda x, y: x * (y + 2), self.grid_2d
        )

        grid_2d_1 = gd.UniformGrid(
            [10, 20], x0=[0.5, 1], dx=[1, 1], ref_level=0
        )

        self.ugd1 = gdu.sample_function_from_uniformgrid(
            lambda x, y: x * (y + 2), grid_2d_1
        )

        grid_2d_2 = gd.UniformGrid(
            [10, 20], x0=[1, 2], dx=[3, 0.4], ref_level=1
        )

        self.ugd2 = gdu.sample_function_from_uniformgrid(
            lambda x, y: x * (y + 2), grid_2d_2
        )

        self.hg = gd.HierarchicalGridData([self.ugd1, self.ugd2])

    def test_unary_functions(self):
        # TimeSeries
        def test_ts(name):
            ma_func = getattr(ma, name)
            np_func = getattr(np.ma, name)
            self.assertTrue(
                ma_func(self.TS), ts.TimeSeries(self.TS.x, np_func(self.TS.y))
            )

        def test_ugd(name):
            ma_func = getattr(ma, name)
            np_func = getattr(np.ma, name)
            self.assertTrue(
                ma_func(self.ugd),
                gd.UniformGridData(self.grid_2d, np_func(self.ugd.data)),
            )

        def test_hg(name):
            ma_func = getattr(ma, name)
            np_func = getattr(np.ma, name)
            expected = [
                gd.UniformGridData(self.ugd1.grid, np_func(self.ugd1.data)),
                gd.UniformGridData(self.ugd2.grid, np_func(self.ugd2.data)),
            ]

            self.assertTrue(ma_func(self.hg), expected)

        for f in [
            "sqrt",
            "exp",
            "log",
            "log2",
            "log10",
            "sin",
            "cos",
            "tan",
            "arcsin",
            "arccos",
            "arctan",
            "arccosh",
            "arctanh",
        ]:
            with self.subTest(f=f):
                test_ts(f)
            with self.subTest(f=f):
                test_ugd(f)
            with self.subTest(f=f):
                test_hg(f)

        # Test method not supported
        with self.assertRaises(AttributeError):
            f = ma._MaskedFunction("bob")
            f(self.TS)

        self.assertEqual(str(ma.sqrt), "Masked version of sqrt")

    def test_mask_where(self):
        def test_ts(name, *args):
            np_func = getattr(np.ma, f"masked_{name}")
            # We will edit in place t
            t = self.TS.copy()
            getattr(t, f"mask_{name}")(*args)
            self.assertTrue(
                t, ts.TimeSeries(self.TS.x, np_func(self.TS.y, *args))
            )

        def test_ugd(name, *args):
            np_func = getattr(np.ma, f"masked_{name}")
            # We will edit in place t
            t = self.ugd.copy()
            getattr(t, f"mask_{name}")(*args)
            self.assertTrue(
                t,
                gd.UniformGridData(
                    self.grid_2d, np_func(self.ugd.data, *args)
                ),
            )

        def test_hg(name, *args):
            np_func = getattr(np.ma, f"masked_{name}")
            # We will edit in place t
            t = self.hg.copy()
            getattr(t, f"mask_{name}")(*args)
            expected = [
                gd.UniformGridData(
                    self.ugd1.grid, np_func(self.ugd1.data, *args)
                ),
                gd.UniformGridData(
                    self.ugd2.grid, np_func(self.ugd2.data, *args)
                ),
            ]

            self.assertTrue(t, expected)

        for f, *vals in [
            ("equal", 0.5),
            ("greater", 0.5),
            ("greater_equal", 0.5),
            ("inside", 0.5, 0.75),
            ("invalid",),
            ("not_equal", 0.5),
            ("less", 0.5),
            ("less_equal", 0.5),
            ("outside", 0.5, 0.75),
        ]:
            with self.subTest(f=f):
                test_ts(f, *vals)
            with self.subTest(f=f):
                test_ugd(f, *vals)
            with self.subTest(f=f):
                test_hg(f, *vals)
