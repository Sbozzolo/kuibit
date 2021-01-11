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

"""Tests for kuibit.grid_data_utils
"""

import unittest

import numpy as np

from kuibit import grid_data as gd
from kuibit import grid_data_utils as gdu


class TestGridDataUtils(unittest.TestCase):
    def test_common_bounding_box(self):

        # Test error for not passing a list
        with self.assertRaises(TypeError):
            gdu.common_bounding_box(1)

        # Test error for not passing a list of UniformGrid
        with self.assertRaises(TypeError):
            gdu.common_bounding_box([1, 2])

        geom1 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[3, 5])
        geom2 = gd.UniformGrid([101], x0=[1], x1=[3])

        # Different dimensions
        with self.assertRaises(ValueError):
            gdu.common_bounding_box([geom1, geom2])

        geom3 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5])
        geom4 = gd.UniformGrid([11, 11], x0=[0, -2], x1=[1, 5])

        self.assertCountEqual(
            gdu.common_bounding_box([geom1, geom3, geom4])[0], [0, -2]
        )
        self.assertCountEqual(
            gdu.common_bounding_box([geom1, geom3, geom4])[1], [5, 5]
        )

        # Test that the function returns the same element when called with one
        # element
        self.assertCountEqual(gdu.common_bounding_box([geom1])[0], geom1.x0)
        self.assertCountEqual(gdu.common_bounding_box([geom1])[1], geom1.x1)

        # All the dimensions are different
        geom5 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5])
        geom6 = gd.UniformGrid([21, 121], x0=[-3, -2], x1=[19, 20])

        self.assertCountEqual(
            gdu.common_bounding_box([geom5, geom6])[0], [-3, -2]
        )
        self.assertCountEqual(
            gdu.common_bounding_box([geom5, geom6])[1], [19, 20]
        )

    def test_merge_uniform_grids(self):

        # Test error for not passing a list
        with self.assertRaises(TypeError):
            gdu.merge_uniform_grids(1)

        # Test error for not passing a list of UniformGrid
        with self.assertRaises(TypeError):
            gdu.merge_uniform_grids([1, 2])

        geom1 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[3, 5], ref_level=1)
        geom2 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[10, 5], ref_level=2)

        # Different ref levels
        with self.assertRaises(ValueError):
            gdu.merge_uniform_grids([geom1, geom2])

        geom3 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[10, 5], ref_level=1)

        # Different dx
        with self.assertRaises(ValueError):
            gdu.merge_uniform_grids([geom1, geom3])

        geom4 = gd.UniformGrid(
            [101, 101], x0=[0, -2], dx=geom1.dx, ref_level=1
        )

        expected_geom = gd.UniformGrid(
            [151, 176], x0=[0, -2], x1=[3, 5], dx=geom1.dx, ref_level=1
        )

        self.assertEqual(
            gdu.merge_uniform_grids([geom1, geom4]), expected_geom
        )

    def test_sample_function(self):

        # Test not grid as input
        with self.assertRaises(TypeError):
            gdu.sample_function_from_uniformgrid(np.sin, 0)

        # Test 1d
        geom = gd.UniformGrid(100, x0=0, x1=2 * np.pi)
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

        self.assertEqual(
            gdu.sample_function(np.sin, 100, 0, 2 * np.pi),
            gd.UniformGridData(geom, data),
        )

        # Test with additional arguments
        geom_ref_level = gd.UniformGrid(100, x0=0, x1=2 * np.pi, ref_level=0)
        self.assertEqual(
            gdu.sample_function(np.sin, 100, 0, 2 * np.pi, ref_level=0),
            gd.UniformGridData(geom_ref_level, data),
        )

        # Test 2d
        geom2d = gd.UniformGrid([100, 200], x0=[0, 1], x1=[1, 2])

        def square(x, y):
            return x * y

        # Test function takes too few arguments
        with self.assertRaises(TypeError):
            gdu.sample_function_from_uniformgrid(lambda x: x, geom2d)

        # Test function takes too many arguments
        with self.assertRaises(TypeError):
            gdu.sample_function_from_uniformgrid(square, geom)

        # Test other TypeError
        with self.assertRaises(TypeError):
            gdu.sample_function_from_uniformgrid(np.sin, geom2d)

        data2d = np.vectorize(square)(*geom2d.coordinates(as_same_shape=True))

        self.assertEqual(
            gdu.sample_function(square, [100, 200], [0, 1], [1, 2]),
            gd.UniformGridData(geom2d, data2d),
        )

        self.assertEqual(
            gdu.sample_function_from_uniformgrid(square, geom2d),
            gd.UniformGridData(geom2d, data2d),
        )
