#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola
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

"""Tests for postcactus.grid_data
"""

import unittest

import numpy as np

from postcactus import grid_data as gd


class TestRegularGeometry(unittest.TestCase):
    def test__check_dims(self):

        # Test multidimensional shape
        with self.assertRaises(ValueError):
            gd.RegularGeometry(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

        # Test different len between shape and origin
        with self.assertRaises(ValueError):
            gd.RegularGeometry(np.array([100, 200]), np.array([1, 2, 3]))

    def test_init_getters(self):

        # Test error neither of delta and x1 provided
        with self.assertRaises(ValueError):
            geom = gd.RegularGeometry([101, 101], [1, 1])

        # Test delta
        geom = gd.RegularGeometry([101, 101], [1, 1], x1=[101, 51])

        self.assertTrue(np.allclose(geom.delta, [1, 0.5]))
        self.assertIs(geom.dx, geom.delta)

        # Test x1 and delta given, but incompatible
        with self.assertRaises(ValueError):
            geom = gd.RegularGeometry(
                [101, 51], [1, 1], x1=[4, 4], delta=[1, 1]
            )

        # Test x1
        geom2 = gd.RegularGeometry([101, 101], [1, 1], delta=[1, 0.5])

        self.assertTrue(np.allclose(geom2.x1, [101, 51]))

        # Test num_ghost
        self.assertCountEqual(geom.num_ghost, np.zeros(2))

        geom3 = gd.RegularGeometry(
            [101, 101], [1, 1], delta=[1, 0.5], num_ghost=[3, 3]
        )

        self.assertCountEqual(geom3.num_ghost, 3 * np.ones(2))

        # Test other attributes
        self.assertEqual(geom3.ref_level, -1)
        self.assertEqual(geom3.component, -1)

        geom4 = gd.RegularGeometry(
            [101, 101],
            [1, 1],
            delta=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertEqual(geom4.time, 1)
        self.assertEqual(geom4.iteration, 1)

        # Test properties
        self.assertEqual(geom4.num_dimensions, 2)
        self.assertAlmostEqual(geom4.dv, 0.5)
        self.assertAlmostEqual(geom4.volume, 0.5 * 101 * 101)

        geom5 = gd.RegularGeometry(
            [101, 101, 1],
            [1, 1, 0],
            delta=[1, 0.5, 0],
            num_ghost=[3, 3, 3],
            time=1,
            iteration=1,
        )

        self.assertCountEqual(geom5.extended_dimensions, [True, True, False])
        self.assertEqual(geom5.num_extended_dimensions, 2)

    def test__in__(self):

        # We test __in__ testing contains, which calls in
        geom4 = gd.RegularGeometry(
            [101, 101],
            [1, 1],
            x1=[101, 51],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertTrue(geom4.contains([50, 50]))
        self.assertTrue(geom4.contains([1, 1]))
        self.assertFalse(geom4.contains([1, 0]))
        self.assertFalse(geom4.contains([102, 102]))
        self.assertFalse(geom4.contains([102, 51]))

    def test__str(self):

        geom4 = gd.RegularGeometry(
            [101, 101],
            [1, 1],
            delta=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertIn("Num ghost zones  = [3 3]", geom4.__str__())

    def test_coordinates(self):

        geom4 = gd.RegularGeometry(
            [11, 15],
            [1, 1],
            delta=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        # Test asking for two types of returns
        with self.assertRaises(ValueError):
            geom4.coordinates(as_meshgrid=True, as_1d_arrays=True)

        x = np.linspace(1, 11, 11)
        y = np.linspace(1, 8, 15)

        c0 = geom4.coordinates(as_meshgrid=True)

        X, Y = np.meshgrid(x, y)

        self.assertTrue(np.allclose(c0[0], X))
        self.assertTrue(np.allclose(c0[1], Y))

        c1 = geom4.coordinates(as_1d_arrays=True)

        self.assertTrue(np.allclose(c1[0], x))
        self.assertTrue(np.allclose(c1[1], y))

        c2 = geom4.coordinates()

        self.assertTrue(np.allclose(c2[0][:, 0], x))
        self.assertTrue(np.allclose(c2[1][0, :], y))

    def test__getitem__(self):

        geom4 = gd.RegularGeometry(
            [11, 15],
            [1, 1],
            delta=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        with self.assertRaises(ValueError):
            geom4[1]

        self.assertCountEqual(geom4[1, 3], [2, 2.5])

    def test_flat_dimensions_remove(self):

        geom = gd.RegularGeometry(
            [101, 101, 1],
            [1, 1, 0],
            delta=[1, 0.5, 0],
            num_ghost=[3, 3, 3],
            time=1,
            iteration=1,
        )

        geom.flat_dimensions_remove()

        geom2 = gd.RegularGeometry(
            [101, 101],
            [1, 1],
            delta=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertEqual(geom, geom2)
