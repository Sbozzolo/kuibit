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


class TestUniformGrid(unittest.TestCase):
    def test__check_dims(self):

        # Test multidimensional shape
        with self.assertRaises(ValueError):
            gd.UniformGrid(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

        # Test different len between shape and origin
        with self.assertRaises(ValueError):
            gd.UniformGrid(np.array([100, 200]), np.array([1, 2, 3]))

    def test_init_getters(self):

        # Test error neither of dx and x1 provided
        with self.assertRaises(ValueError):
            geom = gd.UniformGrid([101, 101], [1, 1])

        # Test dx
        geom = gd.UniformGrid([101, 101], [1, 1], x1=[101, 51])

        self.assertTrue(np.allclose(geom.dx, [1, 0.5]))
        self.assertIs(geom.delta, geom.dx)

        # Test x1 and dx given, but incompatible
        with self.assertRaises(ValueError):
            geom = gd.UniformGrid([101, 51], [1, 1], x1=[4, 4], dx=[1, 1])

        # Test x1 not upper corner
        with self.assertRaises(ValueError):
            geom = gd.UniformGrid([101, 51], [1, 1], x1=[-1, -1])

        # Test x1
        geom2 = gd.UniformGrid([101, 101], [1, 1], dx=[1, 0.5])

        self.assertTrue(np.allclose(geom2.x1, [101, 51]))

        # Test num_ghost
        self.assertCountEqual(geom.num_ghost, np.zeros(2))

        geom3 = gd.UniformGrid(
            [101, 101], [1, 1], dx=[1, 0.5], num_ghost=[3, 3]
        )

        self.assertCountEqual(geom3.num_ghost, 3 * np.ones(2))

        # Test other attributes
        self.assertEqual(geom3.ref_level, -1)
        self.assertEqual(geom3.component, -1)

        geom4 = gd.UniformGrid(
            [101, 101],
            x0=[1, 1],
            dx=[1, 0.5],
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

        geom5 = gd.UniformGrid(
            [101, 101, 1],
            x0=[1, 1, 0],
            dx=[1, 0.5, 0],
            num_ghost=[3, 3, 3],
            time=1,
            iteration=1,
        )

        self.assertCountEqual(geom5.extended_dimensions, [True, True, False])
        self.assertEqual(geom5.num_extended_dimensions, 2)

        # Test case with shape with ones and given x1
        geom6 = gd.UniformGrid(
            [101, 101, 1],
            x0=[1, 1, 0],
            x1=[101, 51, 1],
            num_ghost=[3, 3, 3],
            time=1,
            iteration=1,
        )

        self.assertEqual(geom5, geom6)

    def test__in__(self):

        # We test __in__ testing contains, which calls in
        geom4 = gd.UniformGrid(
            [101, 101],
            x0=[1, 1],
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

        geom4 = gd.UniformGrid(
            [101, 101],
            x0=[1, 1],
            dx=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertIn("Num ghost zones  = [3 3]", geom4.__str__())

    def test_coordinates(self):

        geom4 = gd.UniformGrid(
            [11, 15],
            x0=[1, 1],
            dx=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        x = np.linspace(1, 11, 11)
        y = np.linspace(1, 8, 15)

        c0 = geom4.coordinates(as_meshgrid=True)

        X, Y = np.meshgrid(x, y)

        self.assertTrue(np.allclose(c0[0], X))
        self.assertTrue(np.allclose(c0[1], Y))

        c1 = geom4.coordinates()

        self.assertTrue(np.allclose(c1[0], x))
        self.assertTrue(np.allclose(c1[1], y))

    def test__getitem__(self):

        geom4 = gd.UniformGrid(
            [11, 15],
            x0=[1, 1],
            dx=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        with self.assertRaises(ValueError):
            geom4[1]

        self.assertCountEqual(geom4[1, 3], [2, 2.5])

    def test_flat_dimensions_removed(self):

        geom = gd.UniformGrid(
            [101, 101, 1],
            x0=[1, 1, 0],
            dx=[1, 0.5, 0],
            num_ghost=[3, 3, 3],
            time=1,
            iteration=1,
        )

        geom2 = gd.UniformGrid(
            [101, 101],
            x0=[1, 1],
            dx=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertEqual(geom.flat_dimensions_removed(), geom2)

    def test_shifted(self):

        geom = gd.UniformGrid(
            [101, 101],
            x0=[1, 0],
            x1=[3, 10],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        geom2 = gd.UniformGrid(
            [101, 101],
            x0=[3, -2],
            x1=[5, 8],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertEqual(geom.shifted([2, -2]),
                         geom2)

        # Error incompatible dimensions
        with self.assertRaises(ValueError):
            geom.shifted(2)

    def test_copy(self):

        geom = gd.UniformGrid(
            [101, 101, 1],
            x0=[1, 1, 0],
            dx=[1, 0.5, 0],
            num_ghost=[3, 3, 3],
            time=1,
            iteration=1,
        )

        geom2 = geom.copy()

        self.assertEqual(geom, geom2)
        self.assertIsNot(geom, geom2)

    def test_common_bounding_box(self):

        # Test error for not passing a list
        with self.assertRaises(TypeError):
            gd.common_bounding_box(1)

        # Test error for not passing a list of UniformGrid
        with self.assertRaises(TypeError):
            gd.common_bounding_box([1, 2])

        geom1 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[3, 5])
        geom2 = gd.UniformGrid([101], x0=[1], x1=[3])

        # Different dimensions
        with self.assertRaises(ValueError):
            gd.common_bounding_box([geom1, geom2])

        geom3 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5])
        geom4 = gd.UniformGrid([11, 11], x0=[0, -2], x1=[1, 5])

        self.assertCountEqual(
            gd.common_bounding_box([geom1, geom3, geom4])[0], [0, -2]
        )
        self.assertCountEqual(
            gd.common_bounding_box([geom1, geom3, geom4])[1], [5, 5]
        )

        # Test that the function returns the same element when called with one
        # element
        self.assertCountEqual(gd.common_bounding_box([geom1])[0], geom1.x0)
        self.assertCountEqual(gd.common_bounding_box([geom1])[1], geom1.x1)

        # All the dimensions are different
        geom5 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5])
        geom6 = gd.UniformGrid([21, 121], x0=[-3, -2], x1=[19, 20])

        self.assertCountEqual(
            gd.common_bounding_box([geom5, geom6])[0], [-3, -2]
        )
        self.assertCountEqual(
            gd.common_bounding_box([geom5, geom6])[1], [19, 20]
        )

    def test_merge_uniform_grids(self):

        # Test error for not passing a list
        with self.assertRaises(TypeError):
            gd.merge_uniform_grids(1)

        # Test error for not passing a list of UniformGrid
        with self.assertRaises(TypeError):
            gd.merge_uniform_grids([1, 2])

        geom1 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[3, 5], ref_level=1)
        geom2 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[10, 5], ref_level=2)

        # Different ref levels
        with self.assertRaises(ValueError):
            gd.merge_uniform_grids([geom1, geom2])

        geom3 = gd.UniformGrid([101, 101], x0=[1, 1], x1=[10, 5], ref_level=1)

        # Different dx
        with self.assertRaises(ValueError):
            gd.merge_uniform_grids([geom1, geom3])

        geom4 = gd.UniformGrid(
            [101, 101], x0=[0, -2], dx=geom1.dx, ref_level=1
        )

        expected_geom = gd.UniformGrid(
            [151, 176], x0=[0, -2], x1=[3, 5], dx=geom1.dx, ref_level=1
        )

        self.assertEqual(gd.merge_uniform_grids([geom1, geom4]), expected_geom)

    def test__eq__(self):

        # The tricky part is the time and iteration
        geom0 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5])
        geom1 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5])

        self.assertEqual(geom0, geom1)

        geom2 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5], time=1)

        self.assertNotEqual(geom0, geom2)

        geom3 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5], time=1)

        self.assertEqual(geom3, geom2)

        geom4 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5], iteration=1)

        self.assertNotEqual(geom0, geom4)

        geom5 = gd.UniformGrid([11, 11], x0=[0, 0], x1=[5, 5], iteration=1)

        self.assertEqual(geom5, geom4)

        self.assertNotEqual(geom5, 2)


class TestUniformGridData(unittest.TestCase):
    def setUp(self):
        self.geom = gd.UniformGrid([101, 51], x0=[0, 0], x1=[1, 0.5])

    def test_init(self):

        # Test invalid input
        with self.assertRaises(TypeError):
            gd.UniformGridData(1, 0)

        # Test invalid input
        with self.assertRaises(ValueError):
            gd.UniformGridData(self.geom, np.array([2]))

        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertEqual(ug_data.grid, self.geom)
        self.assertIsNot(ug_data.grid, self.geom)

        self.assertTrue(np.array_equal(ug_data.data, data))
        self.assertIsNot(ug_data.data, data)

        # Test from_grid_structure
        ug_data_from_grid_structure = gd.UniformGridData.from_grid_structure(
            data, x0=[0, 0], x1=[1, 0.5]
        )

        self.assertEqual(ug_data, ug_data_from_grid_structure)

        # Test not equal of UniformGridData
        self.assertNotEqual(ug_data, 2)

        # Test num_dimensions
        self.assertEqual(ug_data.num_dimensions, 2)
        self.assertEqual(ug_data.num_extended_dimensions, 2)

    def test_is_complex(self):

        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertFalse(ug_data.is_complex())

        ug_data_c = gd.UniformGridData(self.geom, 1j * data)

        self.assertTrue(ug_data_c.is_complex())

    def test_flat_dimensions_remove(self):

        geom = gd.UniformGrid([101, 1], x0=[0, 0], x1=[1, 0])

        data = np.array([i * np.linspace(1, 5, 1) for i in range(101)])
        ug_data = gd.UniformGridData(geom, data)

        ug_data.flat_dimensions_remove()

        flat_geom = gd.UniformGrid([101], x0=[0], x1=[1])

        self.assertEqual(
            ug_data, gd.UniformGridData(flat_geom, np.linspace(0, 100, 101))
        )

        # Check invalidation of spline
        self.assertTrue(ug_data.invalid_spline)

    def test__apply_reduction(self):

        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertAlmostEqual(ug_data.min(), 0)
        self.assertAlmostEqual(ug_data.max(), 500)

    def test__apply_binary(self):

        data1 = np.array([i * np.linspace(1, 5, 51) for i in range(101)])
        data2 = np.array([i ** 2 * np.linspace(1, 5, 51) for i in range(101)])
        ug_data1 = gd.UniformGridData(self.geom, data1)
        ug_data2 = gd.UniformGridData(self.geom, data2)

        expected_ug_data = gd.UniformGridData(self.geom, data1 + data2)

        self.assertEqual(ug_data1 + ug_data2, expected_ug_data)

        # Test incompatible grids

        geom = gd.UniformGrid([101, 1], x0=[0, 0], x1=[1, 0])

        data3 = np.array([i * np.linspace(1, 5, 1) for i in range(101)])

        ug_data3 = gd.UniformGridData(geom, data3)

        with self.assertRaises(ValueError):
            ug_data1 + ug_data3

        # Add number
        self.assertEqual(
            ug_data1 + 1, gd.UniformGridData(self.geom, data1 + 1)
        )

        # Incompatible objects
        with self.assertRaises(TypeError):
            ug_data1 + geom

    def test__apply_unary(self):

        data1 = np.array([i * np.linspace(1, 5, 51) for i in range(101)])
        ug_data1 = gd.UniformGridData(self.geom, data1)

        self.assertEqual(
            np.sin(ug_data1), gd.UniformGridData(self.geom, np.sin(data1))
        )

    def test_sample_function(self):

        # Test not grid as input
        with self.assertRaises(TypeError):
            gd.sample_function_from_uniformgrid(np.sin, 0)

        # Test 1d
        geom = gd.UniformGrid(100, x0=0, x1=2 * np.pi)
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

        self.assertEqual(
            gd.sample_function(np.sin, 0, 2 * np.pi, 100),
            gd.UniformGridData(geom, data),
        )

        # Test 2d
        geom2d = gd.UniformGrid([100, 100], x0=[0, 1], x1=[1, 2])

        def square(x, y):
            return x * y

        data2d = np.vectorize(square)(*geom2d.coordinates(as_meshgrid=True))

        self.assertEqual(
            gd.sample_function(square, [0, 1], [1, 2], [100, 100]),
            gd.UniformGridData(geom2d, data2d),
        )

    def test_splines(self):

        # Let's start with 1d.
        sin_data = gd.sample_function(np.sin, 0, 2 * np.pi, 12000)
        sin_data_complex = sin_data + 1j * sin_data

        # Test unknown ext
        with self.assertRaises(ValueError):
            sin_data.evaluate_with_spline(1, ext=3)

        # Test k!=0!=1
        with self.assertRaises(ValueError):
            sin_data._make_spline(k=3)

        self.assertAlmostEqual(
            sin_data_complex.evaluate_with_spline(np.pi / 3),
            (1 + 1j) * np.sin(np.pi / 3),
        )

        # Test __call__

        self.assertAlmostEqual(
            sin_data_complex(np.pi / 3),
            (1 + 1j) * np.sin(np.pi / 3),
        )

        # Vector input
        self.assertTrue(
            np.allclose(
                sin_data_complex.evaluate_with_spline([np.pi / 3, np.pi / 4]),
                np.array(
                    [
                        (1 + 1j) * np.sin(np.pi / 3),
                        (1 + 1j) * np.sin(np.pi / 4),
                    ]
                ),
            )
        )

        # Now 2d
        def product(x, y):
            return x * y

        prod_data = gd.sample_function(product, [0, 0], [3, 3], [101, 101])
        prod_data_complex = (1 + 1j) * prod_data

        self.assertAlmostEqual(
            prod_data_complex.evaluate_with_spline((2, 2)),
            (1 + 1j) * 4,
        )

        # Vector input
        self.assertTrue(
            np.allclose(
                prod_data_complex.evaluate_with_spline([(1, 1), (2, 2)]),
                np.array([(1 + 1j), (1 + 1j) * 4]),
            )
        )

        # Real data
        self.assertAlmostEqual(
            prod_data.evaluate_with_spline((2, 2)),
            4,
        )

        # Extrapolate outside
        self.assertAlmostEqual(
            prod_data.evaluate_with_spline((20, 20), ext=1), 0
        )

        self.assertAlmostEqual(
            prod_data_complex.evaluate_with_spline((20, 20), ext=1), 0
        )

        self.assertTrue(prod_data_complex.spline_real.bounds_error)
        self.assertTrue(prod_data_complex.spline_imag.bounds_error)

    def test_copy(self):

        sin_data = gd.sample_function(np.sin, 0, 2 * np.pi, 1000)

        sin_data2 = sin_data.copy()

        self.assertEqual(sin_data, sin_data2)
        self.assertIsNot(sin_data.data, sin_data2.data)
        self.assertIsNot(sin_data.grid, sin_data2.grid)

    def test_histogram(self):

        # There should be no reason why the histogram behaves differently for
        # different dimensions, so let's test it with 1d
        sin_data = gd.sample_function(np.sin, 0, 2 * np.pi, 1000)
        sin_data_complex = sin_data + 1j * sin_data

        # Test error weights
        with self.assertRaises(TypeError):
            sin_data.histogram(weights=1)

        # Test error complex
        with self.assertRaises(ValueError):
            sin_data_complex.histogram()

        hist = sin_data.histogram()
        expected_hist = np.histogram(sin_data.data, range=(-1, 1), bins=400)

        self.assertTrue(np.allclose(expected_hist[0], hist[0]))
        self.assertTrue(np.allclose(expected_hist[1], hist[1]))

        # Test with weights
        weights = sin_data.copy()
        weights **= 2

        hist = sin_data.histogram(weights)
        expected_hist = np.histogram(
            sin_data.data, range=(-1, 1), bins=400, weights=weights.data
        )

        self.assertTrue(np.allclose(expected_hist[0], hist[0]))
        self.assertTrue(np.allclose(expected_hist[1], hist[1]))

    def test_percentiles(self):

        # There should be no reason why the histogram behaves differently for
        # different dimensions, so let's test it with 1d
        lin_data = gd.sample_function(lambda x: 1.0 * x, 0, 2 * np.pi, 1000)

        # Scalar input
        self.assertAlmostEqual(lin_data.percentiles(0.5), np.pi)

        # Vector input
        self.assertTrue(
            np.allclose(
                lin_data.percentiles([0.25, 0.5]), np.array([np.pi / 2, np.pi])
            )
        )

        # Not normalized
        self.assertTrue(
            np.allclose(
                lin_data.percentiles([250, 500], relative=False),
                np.array([np.pi / 2, np.pi]),
            )
        )

    def test_mean_integral_norm1_norm2(self):

        data = np.array([i ** 2 * np.linspace(1, 5, 51) for i in range(101)])
        ug_data = gd.UniformGridData(self.geom, data)

        self.assertAlmostEqual(ug_data.integral(), np.sum(data) * self.geom.dv)
        self.assertAlmostEqual(
            ug_data.norm1(), np.sum(np.abs(data)) * self.geom.dv
        )
        self.assertAlmostEqual(
            ug_data.norm2(),
            np.sum(np.abs(data) ** 2 * self.geom.dv) ** 0.5,
        )
        self.assertAlmostEqual(
            ug_data.norm_p(3),
            np.sum(np.abs(data) ** 3 * self.geom.dv) ** (1 / 3),
        )
        self.assertAlmostEqual(ug_data.average(), np.mean(data))
