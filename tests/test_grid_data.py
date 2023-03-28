#!/usr/bin/env python3

# Copyright (C) 2020-2023 Gabriele Bozzola
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

"""Tests for kuibit.grid_data
"""

import os
import unittest

import numpy as np

import kuibit.masks as km
from kuibit import grid_data as gd
from kuibit import grid_data_utils as gdu


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

        np.testing.assert_allclose(geom.dx, [1, 0.5])
        self.assertIs(geom.delta, geom.dx)
        self.assertIs(geom.origin, geom.x0)

        # Test x1 and dx given, but incompatible
        with self.assertRaises(ValueError):
            geom = gd.UniformGrid([101, 51], [1, 1], x1=[4, 4], dx=[1, 1])

        # Test x1 not upper corner
        with self.assertRaises(ValueError):
            geom = gd.UniformGrid([101, 51], [1, 1], x1=[-1, -1])

        # Test x1
        geom2 = gd.UniformGrid([101, 101], [1, 1], dx=[1, 0.5])

        np.testing.assert_allclose(geom2.x1, [101, 51])

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

        # Test lowest and highest vertices
        np.testing.assert_allclose(geom5.lowest_vertex, [0.5, 0.75, 0])
        np.testing.assert_allclose(geom5.highest_vertex, [101.5, 51.25, 0])

        # Test case with shape with ones and given x1
        with self.assertRaises(ValueError):
            gd.UniformGrid(
                [101, 101, 1],
                x0=[1, 1, 0],
                x1=[101, 51, 0],
                num_ghost=[3, 3, 3],
                time=1,
                iteration=1,
            )

    def test_hash(self):
        geom4 = gd.UniformGrid(
            [101, 101],
            x0=[1, 1],
            dx=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        self.assertEqual(
            hash(geom4),
            hash(
                (
                    tuple(geom4.shape),
                    tuple(geom4.x0),
                    tuple(geom4.dx),
                    geom4.ref_level,
                    geom4.component,
                    geom4.time,
                    geom4.iteration,
                )
            ),
        )

    def test_coordinate_to_indices(self):
        geom = gd.UniformGrid([101, 51], x0=[1, 2], dx=[1, 0.5])
        # Scalar input
        self.assertCountEqual(geom.indices_to_coordinates([1, 3]), [2, 3.5])
        self.assertCountEqual(geom.coordinates_to_indices([2, 3.5]), [1, 3])
        # Vector input
        np.testing.assert_allclose(
            geom.indices_to_coordinates([[1, 3], [2, 4]]),
            [[2, 3.5], [3, 4]],
        )
        np.testing.assert_allclose(
            geom.coordinates_to_indices([[2, 3.5], [3, 4]]),
            [[1, 3], [2, 4]],
        )

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

        # Edge cases
        # The upper edge is not included
        self.assertFalse(geom4.contains([101.5, 101.25]))
        self.assertFalse(geom4.contains([101.5, 101]))
        self.assertFalse(geom4.contains([101, 101.25]))
        # The lower is
        self.assertTrue(geom4.contains([0.5, 0.75]))
        self.assertTrue(geom4.contains([0.5, 1]))
        self.assertTrue(geom4.contains([1, 0.75]))

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
            x0=[1, 2],
            dx=[1, 0.5],
            num_ghost=[3, 3],
            time=1,
            iteration=1,
        )

        x = np.linspace(1, 11, 11)
        y = np.linspace(2, 9, 15)

        # Test coordinates_1d
        np.testing.assert_allclose(geom4.coordinates_1d[0], x)
        np.testing.assert_allclose(geom4.coordinates_1d[1], y)

        c0 = geom4.coordinates(as_meshgrid=True)

        X, Y = np.meshgrid(x, y)

        np.testing.assert_allclose(c0[0], X)
        np.testing.assert_allclose(c0[1], Y)

        c1 = geom4.coordinates()

        np.testing.assert_allclose(c1[0], x)
        np.testing.assert_allclose(c1[1], y)

        with self.assertRaises(ValueError):
            geom4.coordinates(as_meshgrid=True, as_same_shape=True)

        # Here the output is a list of coordinates shaped like the array itself
        shaped_array = geom4.coordinates(as_same_shape=True)
        self.assertCountEqual(shaped_array[0].shape, geom4.shape)
        # We check that the first column is the same as the coordinates
        np.testing.assert_allclose(
            shaped_array[0][:, 0], geom4.coordinates()[0]
        )

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

        # Check index outside of the grid
        with self.assertRaises(ValueError):
            geom4[[500, 200]]

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

    def test_flat_ghost_zones_removed(self):
        geom = gd.UniformGrid(
            [101, 101], x0=[1, 1], dx=[1, 0.5], num_ghost=[3, 0]
        )

        geom2 = gd.UniformGrid(
            [95, 101], x0=[4, 1], dx=[1, 0.5], num_ghost=[0, 0]
        )

        self.assertEqual(geom.ghost_zones_removed(), geom2)

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

        self.assertEqual(geom.shifted([2, -2]), geom2)

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

        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        self.ug_masked = gd.UniformGridData(
            self.geom, np.ma.masked_greater(data, 10)
        )

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
        self.assertCountEqual(ug_data.extended_dimensions, [True, True])

    def test_is_complex(self):
        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertFalse(ug_data.is_complex())

        ug_data_c = gd.UniformGridData(self.geom, 1j * data)

        self.assertTrue(ug_data_c.is_complex())

    def test_is_masked(self):
        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertFalse(ug_data.is_masked())

        self.assertTrue(self.ug_masked.is_masked())

    def test_mask(self):
        self.assertTrue(
            np.ma.allequal(self.ug_masked.mask, self.ug_masked.data.mask)
        )

        # No mask
        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertTrue(
            np.ma.allequal(ug_data.mask, np.zeros_like(data, dtype=bool))
        )

    def test_mask_apply(self):
        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        data_masked = np.ma.masked_less(data, 3)

        ug_data_mask = gd.UniformGridData(self.geom, data_masked)

        # The log10 will produce warnings because of the zeros in the data
        ug_data_nomask = gd.UniformGridData(self.geom, data)

        ug_data_nomask.mask_apply(ug_data_mask.mask)

        self.assertEqual(ug_data_mask, ug_data_nomask)

        # Now let's add another mask on top

        data_masked2 = np.ma.masked_less(data, 2)

        ug_data_mask2 = gd.UniformGridData(self.geom, data_masked2)

        ug_data_nomask.mask_apply(ug_data_mask2.mask)

        self.assertEqual(ug_data_mask2, ug_data_nomask)

    def test_flat_dimensions_remove(self):
        geom = gd.UniformGrid([101, 1], x0=[0, 0], dx=[0.01, 1])

        data = np.array([i * np.linspace(1, 5, 1) for i in range(101)])
        ug_data = gd.UniformGridData(geom, data)

        ug_data.flat_dimensions_remove()

        flat_geom = gd.UniformGrid([101], x0=[0], x1=[1])

        self.assertEqual(
            ug_data, gd.UniformGridData(flat_geom, np.linspace(0, 100, 101))
        )

        # Check invalidation of spline
        self.assertTrue(ug_data.invalid_spline)

        # Test from 3D to 2D
        grid_data3d = gdu.sample_function_from_uniformgrid(
            lambda x, y, z: x * (y + 2) * (z + 5),
            gd.UniformGrid([10, 20, 1], x0=[0, 1, 0], dx=[1, 1, 1]),
        )
        grid_2d = gd.UniformGrid([10, 20], x0=[0, 1], dx=[1, 1])

        expected_data2d = gdu.sample_function_from_uniformgrid(
            lambda x, y: x * (y + 2) * (0 + 5), grid_2d
        )

        self.assertEqual(
            grid_data3d.flat_dimensions_removed(), expected_data2d
        )

    def test_partial_differentiate(self):
        geom = gd.UniformGrid([8001, 3], x0=[0, 0], x1=[2 * np.pi, 1])

        sin_wave = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom
        )
        original_sin = sin_wave.copy()

        # Error dimension not found
        with self.assertRaises(ValueError):
            sin_wave.partial_differentiated(5)

        # Error grid too small
        sin_wave_tiny = gdu.sample_function(
            lambda x: np.sin(x), [2], x0=[1], x1=[2]
        )
        with self.assertRaises(ValueError):
            sin_wave_tiny.partial_differentiated(0, accuracy_order=4)

        # Second derivative should still be a -sin
        sin_wave.partial_differentiate(0, order=2)

        np.testing.assert_allclose(
            -sin_wave.data, original_sin.data, atol=1e-3
        )

        # print(original_sin.data)
        # print(-original_sin.partial_differentiated(0, order=2, accuracy_order=4).data)

        # Test fourth oder
        np.testing.assert_allclose(
            -original_sin.partial_differentiated(
                0, order=2, accuracy_order=4
            ).data,
            original_sin.data,
            atol=1e-5,
        )

        gradient = original_sin.gradient(order=2)
        np.testing.assert_allclose(
            -gradient[0].data, original_sin.data, atol=1e-3
        )

        # Masked data
        with self.assertRaises(RuntimeError):
            self.ug_masked.partial_differentiated(0)

        # Accuracy order not available
        with self.assertRaises(NotImplementedError):
            original_sin.partial_differentiated(0, accuracy_order=3)

    def test_ghost_zones_remove(self):
        geom = gd.UniformGrid(
            [101, 201], x0=[0, 0], x1=[100, 200], num_ghost=[1, 3]
        )

        data = np.array([i * np.linspace(0, 200, 201) for i in range(101)])
        ug_data = gd.UniformGridData(geom, data)

        ug_data.ghost_zones_remove()

        expected_data = np.array(
            [i * np.linspace(3, 197, 195) for i in range(1, 100)]
        )
        expected_grid = gd.UniformGrid(
            [99, 195], x0=[1, 3], x1=[99, 197], num_ghost=[0, 0]
        )
        expected_ug_data = gd.UniformGridData(expected_grid, expected_data)

        self.assertEqual(ug_data, expected_ug_data)

        # Check invalidation of spline
        self.assertTrue(ug_data.invalid_spline)

        self.assertCountEqual(ug_data.num_ghost, [0, 0])

        # Check with num_ghost = 0
        ug_data.ghost_zones_remove()
        self.assertEqual(ug_data, ug_data.copy())

    def test_reflection_symmetry_undo(self):
        g_zero = gdu.sample_function(
            lambda x, y: 2 * x + y,
            shape=[6, 2],
            x0=[-2, 1],
            x1=[3, 2],
            time=10,
        )

        # The grid looks like: (This grid contains 0)
        #
        # (-2, 2) -- (-1, 2) -- (0, 2) -- (1, 2) -- (2, 2)  --  (3, 2)
        #    |          |         |          |        |           |
        # (-2, 1) -- (-1, 1) -- (0, 1) -- (1, 1) -- (2, 1)  --  (3, 1)
        #
        # We cannot reflect across the y axis, but we can reflect
        # across the x one
        #
        # The data is
        #
        #  -2  --  0  --  2  --  4  --  6 --  8
        #  -3  -- -1  --  1  --  3  --  5 --  7
        #
        # When we reflect across the x axis, we expect a new (7, 2) array:
        #
        #  8  --  6  --  4  --  2  --  4  --  6 --  8
        #  7  --  5  --  3  --  1  --  3  --  5 --  7

        # Test with grid that does not intersect zero
        with self.assertRaises(ValueError):
            g_zero.reflection_symmetry_undone(1)

        # Test with wrong parity
        with self.assertRaises(ValueError):
            g_zero.reflection_symmetry_undone(0, parity=2)

        # Test with grid that cannot be symmetrized
        with self.assertRaises(ValueError):
            gdu.sample_function(
                lambda x: 2 * x,
                shape=[5],
                x0=[-2],
                x1=[3],
                time=10,
            ).reflection_symmetry_undo(0)

        expected_grid = gd.UniformGrid(
            [7, 2], x0=[-3, 1], x1=g_zero.grid.x1, time=10
        )
        expected_data = np.array(
            [[7, 8], [5, 6], [3, 4], [1, 2], [3, 4], [5, 6], [7, 8]]
        )
        expected_g_zero = gd.UniformGridData(expected_grid, expected_data)

        g_zero.reflection_symmetry_undo(0)

        self.assertEqual(expected_g_zero, g_zero)

        # Now a grid that does not contain zero

        g_no_zero = gdu.sample_function(
            lambda x, y: 2 * x + y,
            shape=[4, 2],
            x0=[-3, 1],
            x1=[3, 2],
            time=10,
        )

        # The grid looks like: (This grid contains 0)
        #
        # (-3, 2) -- (-1, 2) -- (1, 2) -- (3, 2)
        #    |          |         |          |
        # (-3, 1) -- (-1, 1) -- (1, 1) -- (3, 1)
        #
        #
        # The data is
        #
        #  -4  --  0  --  4  --  8
        #  -5  -- -1  --  3  --  7
        #
        # When we reflect across the x axis, we expect a new (4, 2) array:
        #
        #  8  --  4  --  4  --  8
        #  7  --  3  --  3  --  7

        expected_grid = gd.UniformGrid(
            [4, 2], x0=[-3, 1], x1=g_zero.grid.x1, time=10
        )
        expected_data = np.array([[7, 8], [3, 4], [3, 4], [7, 8]])
        expected_g_no_zero = gd.UniformGridData(expected_grid, expected_data)

        g_no_zero.reflection_symmetry_undo(0)

        self.assertEqual(expected_g_no_zero, g_no_zero)

    def test_rotation180_symmetry_undo(self):
        g_zero = gdu.sample_function(
            lambda x, y: 2 * x + y,
            shape=[6, 2],
            x0=[-2, -2],
            x1=[3, 2],
            time=10,
        )

        # The grid looks like: (This grid contains 0)
        #
        # (-2, 2) -- (-1, 2) -- (0, 2) -- (1, 2) -- (2, 2)  --  (3, 2)
        #    |          |         |          |        |           |
        # (-2,-2) -- (-1,-2) -- (0,-2) -- (1,-2) -- (2,-2)  --  (3,-2)

        # The data is
        #
        #  -2  --  0  --  2  --  4  --  6 --  8
        #  -6  -- -4  -- -2  --  0  --  2 --  4
        #
        # When we fill-in across the x axis, we expect a new (7, 2) array:
        #
        #  4  --  2  --  0  --  2  --  4  --  6 --  8
        #  8  --  6  --  4  -- -2  --  0  --  2 --  4

        expected_grid = gd.UniformGrid(
            [7, 2], x0=[-3, -2], x1=g_zero.grid.x1, time=10
        )
        expected_data = np.array(
            [[8, 4], [6, 2], [4, 0], [-2, 2], [0, 4], [2, 6], [4, 8]]
        )
        expected_g_zero = gd.UniformGridData(expected_grid, expected_data)

        g_zero.rotation180_symmetry_undo(dimension=0, plane=(0, 1))

        self.assertEqual(expected_g_zero, g_zero)

        # Test wrong dimension
        with self.assertRaises(ValueError):
            g_zero.rotation180_symmetry_undo(dimension=-1, plane=(-1, 2))

        # Test wrong plane
        with self.assertRaises(ValueError):
            g_zero.rotation180_symmetry_undo(dimension=0, plane=(1, 2))

        # Test wrong plane (dimension)
        with self.assertRaises(ValueError):
            g_zero.rotation180_symmetry_undo(dimension=1, plane=(1, 2, 3))

        # Test grid not symmetric along y
        with self.assertRaises(RuntimeError):
            gdu.sample_function(
                lambda x, y: 2 * x + y,
                shape=[6, 2],
                x0=[-2, -1],
                x1=[3, 2],
                time=10,
            ).rotation180_symmetry_undo(dimension=0, plane=(0, 1))

    def test__apply_reduction(self):
        data = np.array([i * np.linspace(1, 5, 51) for i in range(101)])

        ug_data = gd.UniformGridData(self.geom, data)

        self.assertAlmostEqual(ug_data.min(), 0)
        self.assertAlmostEqual(ug_data.max(), 500)

    def test__apply_binary(self):
        data1 = np.array([i * np.linspace(1, 5, 51) for i in range(101)])
        data2 = np.array([i**2 * np.linspace(1, 5, 51) for i in range(101)])
        ug_data1 = gd.UniformGridData(self.geom, data1)
        ug_data2 = gd.UniformGridData(self.geom, data2)

        expected_ug_data = gd.UniformGridData(self.geom, data1 + data2)

        self.assertEqual(ug_data1 + ug_data2, expected_ug_data)

        # Test incompatible grids

        geom = gd.UniformGrid([101, 1], x0=[0, 0], dx=[1, 1])

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

    def test_slice(self):
        grid_data = gdu.sample_function_from_uniformgrid(
            lambda x, y, z: x * (y + 2) * (z + 5),
            gd.UniformGrid([10, 20, 30], x0=[0, 1, 2], dx=[1, 2, 0.1]),
        )
        grid_data_copied = grid_data.copy()

        # Test cut is wrong dimension
        with self.assertRaises(ValueError):
            grid_data.slice([1, 2])

        # Test no cut
        grid_data.slice([None, None, None])
        self.assertEqual(grid_data, grid_data_copied)

        # Test cut point outside the grid
        with self.assertRaises(ValueError):
            grid_data.slice([1, 2, 1000])

        # Test resample

        # Test cut along one dimension
        grid_data.slice([None, None, 3], resample=True)
        expected_no_z = gdu.sample_function_from_uniformgrid(
            lambda x, y: x * (y + 2) * (3 + 5),
            gd.UniformGrid([10, 20], x0=[0, 1], dx=[1, 2]),
        )
        self.assertEqual(grid_data, expected_no_z)

        # Test no resample
        #
        # Reset grid data
        grid_data = grid_data_copied.copy()
        grid_data.slice([None, None, 3], resample=False)
        self.assertEqual(grid_data, expected_no_z)

    def test_coordinates_at(self):
        grid_2d = gd.UniformGrid([10, 20], x0=[1, 2], dx=[1, 1])

        data2d = gdu.sample_function_from_uniformgrid(
            lambda x, y: -x * (y + 2), grid_2d
        )

        expected = [1, 2]

        np.testing.assert_allclose(
            expected, data2d.coordinates_at_maximum(absolute=False)
        )
        np.testing.assert_allclose(expected, data2d.coordinates_at_minimum())

    def test_save_load(self):
        grid_file = "test_save_grid.dat"
        grid_file_bz = "test_save_grid.dat.bz2"
        grid_file_gz = "test_save_grid.dat.gz"

        grid_file_npz = "test_save_grid.dat.npz"

        def square(x, y):
            return x * y

        grid_data = gdu.sample_function(square, [100, 200], [0, 1], [1, 2])

        # Test save uncompressed
        grid_data.save(grid_file)

        # Load it
        loaded = gdu.load_UniformGridData(grid_file)

        self.assertEqual(loaded, grid_data)

        # Clean up file
        os.remove(grid_file)

        # Test compressed
        grid_data.save(grid_file_bz)
        loaded_bz = gdu.load_UniformGridData(grid_file_bz)

        self.assertEqual(loaded_bz, grid_data)

        # Clean up file
        os.remove(grid_file_bz)

        grid_data.save(grid_file_gz)
        loaded_gz = gdu.load_UniformGridData(grid_file_gz)

        self.assertEqual(loaded_gz, grid_data)

        # Clean up file
        os.remove(grid_file_gz)

        # Test npz
        grid_data.save(grid_file_npz)
        loaded_npz = gdu.load_UniformGridData(grid_file_npz)

        self.assertEqual(loaded_npz, grid_data)

        # Clean up file
        os.remove(grid_file_npz)

        # Now let's consider a case with also time and iteration. These are
        # treated differently
        grid_file_npz_ti = "test_save_grid_ti.dat.npz"

        grid_data_ti = gdu.sample_function(
            square, [100, 200], [0, 1], [1, 2], time=1, iteration=2
        )

        grid_data_ti.save(grid_file_npz_ti)
        loaded_npz_ti = gdu.load_UniformGridData(grid_file_npz_ti)

        self.assertEqual(loaded_npz_ti, grid_data_ti)

        # Clean up file
        os.remove(grid_file_npz_ti)

        with self.assertWarns(RuntimeWarning):
            path = "/tmp/tmp_kuibit.dat"
            self.ug_masked.save(path)
            os.remove(path)

    def test_splines(self):
        # Let's start with 1d.
        sin_data = gdu.sample_function(np.sin, 12000, 0, 2 * np.pi)
        sin_data_complex = sin_data + 1j * sin_data

        # Test unknown ext
        with self.assertRaises(ValueError):
            sin_data.evaluate_with_spline(1, ext=3)

        # Test k!=0!=1
        with self.assertRaises(ValueError):
            sin_data._make_spline(k=3)

        self.assertAlmostEqual(
            sin_data_complex.evaluate_with_spline([np.pi / 3]),
            (1 + 1j) * np.sin(np.pi / 3),
        )

        # Test with point in cell but outside boundary, in

        # We change the boundary values to be different from 0
        sin_data_complex_plus_one = sin_data_complex + 1 + 1j

        dx = sin_data_complex.dx[0]
        # At the boundary, we do a constant extrapolation, so the value should
        # be the boundary value
        self.assertAlmostEqual(
            sin_data_complex_plus_one.evaluate_with_spline([0 - 0.25 * dx]),
            (1 + 1j),
        )
        self.assertAlmostEqual(
            sin_data_complex_plus_one.evaluate_with_spline(
                [2 * np.pi + 0.25 * dx]
            ),
            (1 + 1j),
        )

        # Test __call__
        self.assertAlmostEqual(
            sin_data_complex([np.pi / 3]),
            (1 + 1j) * np.sin(np.pi / 3),
        )

        # Test on a point of the grid
        point = [sin_data.grid.coordinates_1d[0][2]]
        self.assertAlmostEqual(
            sin_data_complex(point),
            (1 + 1j) * np.sin(point[0]),
        )

        # Test on a point outside the grid with the lookup table
        with self.assertRaises(ValueError):
            sin_data_complex._nearest_neighbor_interpolation(
                np.array([1000]),
                ext=2,
            )
        # Test on a point outside the grid with the lookup table and
        # ext = 1
        self.assertEqual(
            sin_data_complex.evaluate_with_spline(
                [1000], ext=1, piecewise_constant=True
            ),
            0,
        )

        # Vector input
        np.testing.assert_allclose(
            sin_data_complex.evaluate_with_spline([[np.pi / 3], [np.pi / 4]]),
            np.array(
                [
                    (1 + 1j) * np.sin(np.pi / 3),
                    (1 + 1j) * np.sin(np.pi / 4),
                ]
            ),
        )

        # Vector input in, vector input out
        self.assertEqual(sin_data_complex([[1]]).shape, (1,))

        # Now 2d
        def product(x, y):
            return x * (y + 2)

        prod_data = gdu.sample_function(product, [101, 101], [0, 0], [3, 3])
        prod_data_complex = (1 + 1j) * prod_data

        self.assertAlmostEqual(
            prod_data_complex.evaluate_with_spline((2, 3)),
            (1 + 1j) * 10,
        )

        # Vector input
        np.testing.assert_allclose(
            prod_data_complex.evaluate_with_spline([(1, 0), (2, 3)]),
            np.array([(1 + 1j) * 2, (1 + 1j) * 10]),
        )

        np.testing.assert_allclose(
            prod_data_complex.evaluate_with_spline(
                [[(1, 0), (2, 3)], [(3, 1), (0, 0)]]
            ),
            np.array([[(1 + 1j) * 2, (1 + 1j) * 10], [(1 + 1j) * 9, 0]]),
        )

        # Real data
        self.assertAlmostEqual(
            prod_data.evaluate_with_spline((2, 3)),
            10,
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

        # Test on a UniformGrid
        sin_data = gdu.sample_function(np.sin, 12000, 0, 2 * np.pi)
        linspace = gd.UniformGrid(101, x0=0, x1=3)
        output = sin_data(linspace)
        np.testing.assert_allclose(
            output.data, np.sin(linspace.coordinates()[0])
        )

        # Incompatible dimensions
        with self.assertRaises(ValueError):
            sin_data(gd.UniformGrid([101, 201], x0=[0, 1], x1=[3, 4]))

        # Test with grid that has a flat dimension
        prod_data_flat = gdu.sample_function_from_uniformgrid(
            product, gd.UniformGrid([101, 1], x0=[0, 0], dx=[1, 3])
        )

        # y = 1 is in the flat cell, where y = 0, and here we are using nearest
        # interpolation
        self.assertAlmostEqual(prod_data_flat((1, 1)), 2)
        # Vector
        self.assertCountEqual(prod_data_flat([(1, 1), (2, 1)]), [2, 4])

        # Masked data
        two_points = gdu.sample_function_from_uniformgrid(
            lambda x, y: 1, gd.UniformGrid([2, 2], x0=[0, 0], dx=[1, 1])
        )

        two_points.data = np.ma.MaskedArray(
            two_points.data, mask=[[True, True], [False, False]]
        )

        expected_out = np.ma.MaskedArray([1], mask=[True])

        # One point
        self.assertEqual(
            two_points._nearest_neighbor_interpolation([[0.25, 0.25]]),
            expected_out,
        )

        expected_out = np.ma.MaskedArray([1, 1], mask=[True, False])

        # Two points
        self.assertTrue(
            np.ma.allequal(
                two_points._nearest_neighbor_interpolation(
                    [[0.25, 0.25], [0.8, 0.8]]
                ),
                expected_out,
            )
        )

        with self.assertRaises(RuntimeError):
            two_points._make_spline()

    def test_copy(self):
        sin_data = gdu.sample_function(np.sin, 1000, 0, 2 * np.pi)

        sin_data2 = sin_data.copy()

        self.assertEqual(sin_data, sin_data2)
        self.assertIsNot(sin_data.data, sin_data2.data)
        self.assertIsNot(sin_data.grid, sin_data2.grid)

    def test_histogram(self):
        # There should be no reason why the histogram behaves differently for
        # different dimensions, so let's test it with 1d
        sin_data = gdu.sample_function(np.sin, 1000, 0, 2 * np.pi)
        sin_data_complex = sin_data + 1j * sin_data

        # Test error weights
        with self.assertRaises(TypeError):
            sin_data.histogram(weights=1)

        # Test error complex
        with self.assertRaises(ValueError):
            sin_data_complex.histogram()

        hist = sin_data.histogram()
        expected_hist = np.histogram(sin_data.data, range=(-1, 1), bins=400)

        np.testing.assert_allclose(expected_hist[0], hist[0], rtol=2e-6)
        np.testing.assert_allclose(expected_hist[1], hist[1], rtol=2e-6)

        # Test with weights
        weights = sin_data.copy()
        weights **= 2

        hist = sin_data.histogram(weights)
        expected_hist = np.histogram(
            sin_data.data, range=(-1, 1), bins=400, weights=weights.data
        )

        np.testing.assert_allclose(expected_hist[0], hist[0], rtol=2e-6)
        np.testing.assert_allclose(expected_hist[1], hist[1], rtol=2e-6)

    def test_percentiles(self):
        # There should be no reason why the histogram behaves differently for
        # different dimensions, so let's test it with 1d
        lin_data = gdu.sample_function(lambda x: 1.0 * x, 1000, 0, 2 * np.pi)

        # Scalar input
        self.assertAlmostEqual(lin_data.percentiles(0.5), np.pi)

        # Vector input
        np.testing.assert_allclose(
            lin_data.percentiles([0.25, 0.5]), np.array([np.pi / 2, np.pi])
        )

        # Not normalized
        np.testing.assert_allclose(
            lin_data.percentiles([250, 500], relative=False),
            np.array([np.pi / 2, np.pi]),
        )

    def test_mean_integral_norm1_norm2(self):
        data = np.array([i**2 * np.linspace(1, 5, 51) for i in range(101)])
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

    def test_resampled(self):
        def product(x, y):
            return x * (y + 2)

        def product_complex(x, y):
            return (1 + 1j) * x * (y + 2)

        prod_data = gdu.sample_function(product, [101, 201], [0, 1], [3, 4])
        prod_data_complex = gdu.sample_function(
            product_complex, [3001, 2801], [0, 1], [3, 4]
        )
        # Check error
        with self.assertRaises(TypeError):
            prod_data.resampled(2)

        # Check same grid
        self.assertEqual(prod_data.resampled(prod_data.grid), prod_data)

        new_grid = gd.UniformGrid([51, 101], x0=[1, 2], x1=[2, 3])

        resampled = prod_data_complex.resampled(new_grid)
        exp_resampled = gdu.sample_function_from_uniformgrid(
            product_complex, new_grid
        )

        self.assertEqual(resampled.grid, new_grid)
        np.testing.assert_allclose(resampled.data, exp_resampled.data)

        # Check that the method of the spline is linear
        self.assertEqual(prod_data_complex.spline_imag.method, "linear")

        # Test using nearest interpolation
        resampled_nearest = prod_data_complex.resampled(
            new_grid, piecewise_constant=True
        )

        np.testing.assert_allclose(
            resampled_nearest.data, exp_resampled.data, rtol=1e-3
        )

        # Check that the method of the spline hasn't linear
        self.assertEqual(prod_data_complex.spline_imag.method, "linear")

        # Check single number
        self.assertAlmostEqual(resampled_nearest((2, 2.5)), 9 * (1 + 1j))

        # Check with one point
        new_grid2 = gd.UniformGrid([11, 1], x0=[1, 2], dx=[0.1, 1])
        resampled2 = prod_data_complex.resampled(new_grid2)
        prod_data_one_point = gdu.sample_function_from_uniformgrid(
            product_complex, new_grid2
        )

        self.assertEqual(resampled2, prod_data_one_point)

        # Resample from 3d to 2d

        grid_data3d = gdu.sample_function_from_uniformgrid(
            lambda x, y, z: x * (y + 2) * (z + 5),
            gd.UniformGrid([10, 20, 11], x0=[0, 1, 0], dx=[1, 2, 0.1]),
        )
        grid_2d = gd.UniformGrid([10, 20, 1], [0, 1, 0], dx=[1, 2, 0.1])

        expected_data2d = gdu.sample_function_from_uniformgrid(
            lambda x, y, z: x * (y + 2) * (z + 5), grid_2d
        )

        self.assertEqual(grid_data3d.resampled(grid_2d), expected_data2d)

    def test_dx_change(self):
        def product_complex(x, y):
            return (1 + 1j) * x * (y + 2)

        prod_data_complex = gdu.sample_function(
            product_complex, [301, 401], [0, 1], [3, 4]
        )

        prod_data_complex_copy = prod_data_complex.copy()

        # Test invalid new dx
        # Not a list
        with self.assertRaises(TypeError):
            prod_data_complex.dx_change(0)

        # Not a with the correct dimensions
        with self.assertRaises(ValueError):
            prod_data_complex.dx_change([0])

        # Not a integer multiple/factor
        with self.assertRaises(ValueError):
            prod_data_complex.dx_change(prod_data_complex.dx * np.pi)

        # Same dx
        self.assertEqual(
            prod_data_complex.dx_changed(prod_data_complex.dx),
            prod_data_complex,
        )

        # Half dx
        prod_data_complex.dx_change(prod_data_complex.dx / 2)
        self.assertCountEqual(
            prod_data_complex.dx, prod_data_complex_copy.dx / 2
        )
        self.assertCountEqual(
            prod_data_complex.shape, prod_data_complex_copy.shape * 2 - 1
        )
        # The data part should be tested with testing resample

        # Twice of the dx, which will bring us back to same dx,
        # so, same object we started with
        prod_data_complex.dx_change(prod_data_complex.dx * 2)
        self.assertEqual(prod_data_complex, prod_data_complex_copy)

    def test_coordinates(self):
        def square(x, y):
            return x * (y + 2)

        grid_data = gdu.sample_function_from_uniformgrid(square, self.geom)

        np.testing.assert_allclose(
            grid_data.coordinates_from_grid()[0],
            self.geom.coordinates()[0],
        )

        # This is a list of UniformGridData
        grids = grid_data.coordinates()
        # Here we check that they agree on two coordinates
        for dim in range(len(grids)):
            self.assertAlmostEqual(
                grids[dim](self.geom[2, 3]), self.geom[2, 3][dim]
            )

        # Here we test coordiantes_meshgrid()
        np.testing.assert_allclose(
            grid_data.coordinates_meshgrid()[0][0], self.geom.coordinates()[0]
        )

    def test_properties(self):
        def square(x, y):
            return x * (y + 2)

        grid_data = gdu.sample_function_from_uniformgrid(square, self.geom)

        self.assertCountEqual(grid_data.x0, self.geom.x0)
        self.assertCountEqual(grid_data.origin, self.geom.x0)
        self.assertCountEqual(grid_data.shape, self.geom.shape)
        self.assertCountEqual(grid_data.x1, self.geom.x1)
        self.assertCountEqual(grid_data.dx, self.geom.dx)
        self.assertCountEqual(grid_data.delta, self.geom.dx)
        self.assertCountEqual(grid_data.num_ghost, self.geom.num_ghost)
        self.assertEqual(grid_data.ref_level, self.geom.ref_level)
        self.assertEqual(grid_data.component, self.geom.component)
        self.assertEqual(grid_data.time, self.geom.time)
        self.assertEqual(grid_data.iteration, self.geom.iteration)
        np.testing.assert_allclose(grid_data.data_xyz, grid_data.data.T)

    def test__getitem__(self):
        def square(x, y):
            return x * (y + 2)

        # These are just integers
        prod_data = gdu.sample_function(square, [11, 21], [0, 10], [10, 30])

        self.assertAlmostEqual(prod_data[2, 2], 2 * 14)

    def test_fourier_transform(self):
        prod_data_complex = gdu.sample_function(
            lambda x, y: (1 + 1j) * x * (y + 2), [11, 21], [0, 10], [10, 30]
        )

        dx = prod_data_complex.dx
        fft_c = np.fft.fftshift(np.fft.fftn(prod_data_complex.data))
        freqs_c = [
            np.fft.fftshift(np.fft.fftfreq(11, d=dx[0])),
            np.fft.fftshift(np.fft.fftfreq(21, d=dx[1])),
        ]
        f_min_c = [freqs_c[0][0], freqs_c[1][0]]
        delta_f_c = [
            freqs_c[0][1] - freqs_c[0][0],
            freqs_c[1][1] - freqs_c[1][0],
        ]

        freq_grid_c = gd.UniformGrid(fft_c.shape, x0=f_min_c, dx=delta_f_c)
        expected_c = gd.UniformGridData(freq_grid_c, fft_c)

        self.assertEqual(expected_c, prod_data_complex.fourier_transform())

        # Masked data
        with self.assertRaises(RuntimeError):
            self.ug_masked.fourier_transform()

    def test_to_GridSeries(self):
        # Not 1D
        prod_data_complex = gdu.sample_function(
            lambda x, y: (1 + 1j) * x * (y + 2), [11, 21], [0, 10], [10, 30]
        )
        with self.assertRaises(ValueError):
            prod_data_complex.to_GridSeries()

        prod_data_complex_1D = gdu.sample_function(
            lambda x: (1 + 1j) * x, [11], [0], [10]
        )

        coords = np.linspace(0, 10, 11)

        expected_gridseries = gd.GridSeries(coords, (1 + 1j) * coords)

        self.assertEqual(
            prod_data_complex_1D.to_GridSeries(), expected_gridseries
        )


class TestHierarchicalGridData(unittest.TestCase):
    def setUp(self):
        # Here we split the rectangle with x0 = [0, 1], x1 = [14, 26]
        # and shape [14, 26] in 4 pieces
        patch1 = gd.UniformGrid([4, 5], x0=[0, 1], x1=[3, 5], ref_level=0)
        patch2 = gd.UniformGrid([11, 21], x0=[4, 6], x1=[14, 26], ref_level=0)
        patch3 = gd.UniformGrid([11, 5], x0=[4, 1], x1=[14, 5], ref_level=0)
        patch4 = gd.UniformGrid([4, 21], x0=[0, 6], x1=[3, 26], ref_level=0)

        self.grids0 = [patch1, patch2, patch3, patch4]
        # self.grids1 are not to be merged because they do not fill the space
        self.grids1 = [patch1, patch2]

        def product(x, y):
            return x * (y + 2)

        self.grid_data = [
            gdu.sample_function_from_uniformgrid(product, g)
            for g in self.grids0
        ]

        self.grid_data_two_comp = [
            gdu.sample_function_from_uniformgrid(product, g)
            for g in self.grids1
        ]

        self.expected_grid = gd.UniformGrid(
            [15, 26], x0=[0, 1], x1=[14, 26], ref_level=0
        )

        self.expected_data = gdu.sample_function_from_uniformgrid(
            product, self.expected_grid
        )

        # We also consider one grid data with a different refinement level
        self.expected_grid_level2 = gd.UniformGrid(
            [15, 26], x0=[0, 1], x1=[14, 26], ref_level=2
        )

        self.expected_data_level2 = gdu.sample_function_from_uniformgrid(
            product, self.expected_grid_level2
        )

    def test_init(self):
        # Test incorrect arguments
        # Not a list
        with self.assertRaises(TypeError):
            gd.HierarchicalGridData(0)

        # Empty list
        with self.assertRaises(ValueError):
            gd.HierarchicalGridData([])

        # Not a list of UniformGridData
        with self.assertRaises(TypeError):
            gd.HierarchicalGridData([0])

        # Inconsistent number of dimensions
        def product1(x):
            return x

        def product2(x, y):
            return x * y

        prod_data1 = gdu.sample_function(product1, [101], [0], [3])
        prod_data2 = gdu.sample_function(product2, [101, 101], [0, 0], [3, 3])

        with self.assertRaises(ValueError):
            gd.HierarchicalGridData([prod_data1, prod_data2])

        # Only one component
        one = gd.HierarchicalGridData([prod_data1])
        # Test content
        self.assertDictEqual(
            one.grid_data_dict, {-1: [prod_data1.ghost_zones_removed()]}
        )

        grid = gd.UniformGrid([101], x0=[0], x1=[3], ref_level=2)

        # Two components at two different levels
        prod_data1_level2 = gdu.sample_function_from_uniformgrid(
            product1, grid
        )
        two = gd.HierarchicalGridData([prod_data1, prod_data1_level2])
        self.assertDictEqual(
            two.grid_data_dict,
            {
                -1: [prod_data1.ghost_zones_removed()],
                2: [prod_data1_level2.ghost_zones_removed()],
            },
        )

        # Test a good grid
        hg_many_components = gd.HierarchicalGridData(self.grid_data)
        self.assertEqual(
            hg_many_components.grid_data_dict[0], [self.expected_data]
        )

        # Test a grid with two separate components
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        self.assertEqual(hg3.grid_data_dict[0], self.grid_data_two_comp)

        # Test with merged masked data
        grid_data = self.grid_data[:]
        # Make one of the data Masked, we will check that the entire
        # data is masked
        grid_data[0].data = np.ma.MaskedArray(grid_data[0].data)
        hg_merged = gd.HierarchicalGridData(grid_data)

        self.assertTrue(isinstance(hg_merged[0][0].data, np.ma.MaskedArray))

    def test_check_ref_factors(self):
        # Check a good grid, with refinement factors that are a constant
        # multiple of the finest refinement level.
        fine = gdu.sample_function(
            lambda x, y: x * y, [101, 101], [0, 0], [3, 3], ref_level=3
        )
        coarse = gdu.sample_function(
            lambda x, y: x * y, [51, 51], [0, 0], [3, 3], ref_level=2
        )
        very_coarse = gdu.sample_function(
            lambda x, y: x * y, [26, 26], [0, 0], [3, 3], ref_level=1
        )

        self.assertTrue(
            gd.HierarchicalGridData(
                [fine, coarse, very_coarse]
            )._check_ref_factors()
        )

        # Now a case with non integer refinement factors.
        almost_coarse = gdu.sample_function(
            lambda x, y: x * y, [56, 56], [0, 0], [3, 3], ref_level=2
        )
        self.assertFalse(
            gd.HierarchicalGridData([fine, almost_coarse])._check_ref_factors()
        )

        # Finally a case with non constant
        very_fine = gdu.sample_function(
            lambda x, y: x * y, [401, 401], [0, 0], [3, 3], ref_level=4
        )
        self.assertFalse(
            gd.HierarchicalGridData(
                [very_fine, fine, coarse]
            )._check_ref_factors()
        )

        # Check case with only one refinement level
        self.assertTrue(
            gd.HierarchicalGridData([very_fine])._check_ref_factors()
        )

    def test__getitem__(self):
        hg = gd.HierarchicalGridData(self.grid_data)
        self.assertEqual(hg[0], [self.expected_data])

    def test_get_level(self):
        hg = gd.HierarchicalGridData(self.grid_data)
        self.assertEqual(hg.get_level(0), self.expected_data)

        # Level not available
        with self.assertRaises(ValueError):
            hg.get_level(10)

        # Multiple patches will throw an error
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        with self.assertRaises(ValueError):
            hg3.get_level(0)

    def test_shape(self):
        hg = gd.HierarchicalGridData(self.grid_data)
        self.assertCountEqual(hg.shape, {0: 1})

        # Multiple patches will throw an error
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        self.assertCountEqual(hg3.shape, {0: 2})

    def test_properties(self):
        # len
        hg = gd.HierarchicalGridData(
            self.grid_data + [self.expected_data_level2]
        )
        self.assertEqual(len(hg), 2)

        # refinement levels
        self.assertCountEqual(hg.refinement_levels, [0, 2])

        # grid_data
        self.assertCountEqual(
            hg.all_components, [self.expected_data, self.expected_data_level2]
        )

        # first component
        self.assertEqual(hg.first_component, hg[0][0])

        # finest level
        self.assertEqual(hg.num_finest_level, 2)

        # max refinement_level
        self.assertEqual(hg.max_refinement_level, 2)

        # coarsest level
        self.assertEqual(hg.num_coarsest_level, 0)

        # dtype
        self.assertEqual(hg.dtype, float)

        # x0, x1
        self.assertCountEqual(hg.x0, self.expected_data.x0)
        self.assertCountEqual(hg.x1, self.expected_data.x1)
        # For multiple components there should be an error
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        with self.assertRaises(ValueError):
            hg3.x0
        with self.assertRaises(ValueError):
            hg3.x1

        # dx_at_level, dx coarsest, fines
        self.assertCountEqual(hg.dx_at_level(0), [1, 1])
        self.assertCountEqual(hg3.dx_at_level(0), [1, 1])
        self.assertCountEqual(hg.coarsest_dx, [1, 1])
        self.assertCountEqual(hg.finest_dx, [1, 1])

        # num dimensions
        self.assertEqual(hg.num_dimensions, 2)
        self.assertEqual(hg.num_extended_dimensions, 2)

        # time and iteration
        self.assertIs(hg.time, None)
        self.assertIs(hg.iteration, None)

    def test__eq__(self):
        hg1 = gd.HierarchicalGridData(self.grid_data)
        hg2 = gd.HierarchicalGridData([self.expected_data_level2])
        hg3 = gd.HierarchicalGridData([self.expected_data])

        self.assertNotEqual(hg1, hg2)
        self.assertEqual(hg1, hg3)

        # Not same type
        self.assertNotEqual(hg1, 2)

        hg4 = gd.HierarchicalGridData(
            [self.expected_data, self.expected_data_level2]
        )
        # Not same number of refinement levels
        self.assertNotEqual(hg1, hg4)

        # Multiple components
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        self.assertEqual(hg3, hg3)

    def test_copy(self):
        hg1 = gd.HierarchicalGridData(self.grid_data)
        hg2 = hg1.copy()
        self.assertEqual(hg1, hg2)
        self.assertIsNot(hg1, hg2)

        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        hg4 = hg3.copy()
        self.assertEqual(hg3, hg4)

    def test_is_complex(self):
        hg_real = gd.HierarchicalGridData(self.grid_data_two_comp)

        self.assertFalse(hg_real.is_complex())

        hg_complex = gd.HierarchicalGridData(self.grid_data_two_comp)

        # Make it complex
        hg_complex[0][0] *= 1j

        self.assertTrue(hg_complex.is_complex())

    def test_is_masked(self):
        hg = gd.HierarchicalGridData(self.grid_data_two_comp)

        self.assertFalse(hg.is_masked())

        hg_masked = km.arcsin(gd.HierarchicalGridData(self.grid_data_two_comp))

        self.assertTrue(hg_masked.is_masked())

        # Test mask

        self.assertTrue(
            np.ma.allequal(hg_masked.mask[0], hg_masked[0][0].data.mask)
        )

        # Test apply_mask
        hg_nomasked = gd.HierarchicalGridData(self.grid_data_two_comp)
        hg_nomasked.mask_apply(hg_masked.mask)

        self.assertEqual(hg_nomasked, hg_masked)

    def test_iter(self):
        hg1 = gd.HierarchicalGridData(self.grid_data)

        for ref_level, comp, data in hg1:
            self.assertTrue(isinstance(data, gd.UniformGridData))
            self.assertEqual(ref_level, 0)
            self.assertEqual(comp, 0)

        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)

        comp_index = 0
        for ref_level, comp, data in hg3:
            self.assertEqual(ref_level, 0)
            self.assertEqual(comp, comp_index)
            self.assertTrue(isinstance(data, gd.UniformGridData))
            comp_index += 1

        # Test from finest
        geom = gd.UniformGrid(
            [81, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=0
        )
        geom2 = gd.UniformGrid(
            [11, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=1
        )

        sin_wave1 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom
        )
        sin_wave2 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom2
        )

        sin_wave = gd.HierarchicalGridData([sin_wave1] + [sin_wave2])

        index = 1
        for ref_level, comp, data in sin_wave.iter_from_finest():
            self.assertEqual(ref_level, index)
            self.assertEqual(comp, 0)
            self.assertTrue(isinstance(data, gd.UniformGridData))
            index -= 1

    def test_finest_coarsest_level(self):
        geom = gd.UniformGrid(
            [81, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=0
        )
        geom2 = gd.UniformGrid(
            [11, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=1
        )

        sin_wave1 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom
        )
        sin_wave2 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom2
        )

        sin_wave = gd.HierarchicalGridData([sin_wave1] + [sin_wave2])

        self.assertEqual(sin_wave.finest_level, sin_wave2)
        self.assertEqual(sin_wave.coarsest_level, sin_wave1)

    def test__apply_reduction(self):
        hg1 = gd.HierarchicalGridData(self.grid_data)

        self.assertAlmostEqual(hg1.min(), 0)

        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)

        self.assertAlmostEqual(hg3.min(), 0)

    def test__apply_unary(self):
        hg1 = gd.HierarchicalGridData(self.grid_data)

        def neg_product(x, y):
            return -x * (y + 2)

        neg_data = gdu.sample_function_from_uniformgrid(
            neg_product, self.expected_grid
        )

        hg2 = gd.HierarchicalGridData([neg_data])

        self.assertEqual(-hg1, hg2)

        # Test with multiple components
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)

        hg4 = hg3.copy()
        hg4[0][0] *= -1
        hg4[0][1] *= -1

        self.assertEqual(-hg3, hg4)

    def test__apply_binary(self):
        hg1 = gd.HierarchicalGridData(self.grid_data)

        # Test incompatible types
        with self.assertRaises(TypeError):
            hg1 + "hey"

        def neg_product(x, y):
            return -x * (y + 2)

        neg_data = gdu.sample_function_from_uniformgrid(
            neg_product, self.expected_grid
        )

        hg2 = gd.HierarchicalGridData([neg_data])

        zero = hg1 + hg2
        zero += 0

        # To check that zero is indeed zero we check that the abs max of the
        # data is 0
        self.assertEqual(np.amax(np.abs(zero[0][0].data)), 0)

        # Test incompatible refinement levels

        neg_data_level2 = gdu.sample_function_from_uniformgrid(
            neg_product, self.expected_grid_level2
        )

        with self.assertRaises(ValueError):
            hg1 + gd.HierarchicalGridData([neg_data_level2])

        # Test with multiple components
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)

        hg4 = hg3.copy()
        hg4[0][0] *= -1
        hg4[0][1] *= -1

        zero2 = hg3 + hg4
        self.assertEqual(np.amax(np.abs(zero2[0][0].data)), 0)
        self.assertEqual(np.amax(np.abs(zero2[0][1].data)), 0)

    def test_finest_component_at_point(self):
        # Using the component mapping

        hg = gd.HierarchicalGridData(
            self.grid_data + [self.expected_data_level2]
        )

        # Input is not a valid point
        with self.assertRaises(TypeError):
            hg.finest_component_at_point(0)

        # Dimensionality mismatch
        with self.assertRaises(ValueError):
            hg.finest_component_at_point([0])

        # Point outside the grid
        with self.assertRaises(ValueError):
            hg.finest_component_at_point([1000, 200])

        self.assertEqual(hg.finest_component_at_point([3, 4]), hg[2][0])

        # Test with multiple components
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)
        self.assertEqual(hg3.finest_component_at_point([3, 4]), hg3[0][0])
        # Test on edge of the two components
        self.assertEqual(hg3.finest_component_at_point([3, 5]), hg3[0][0])
        self.assertEqual(hg3.finest_component_at_point([4, 6]), hg3[0][1])

        # Using the general method. The general method kicks in when the
        # refinement levels do not have integral refinement factors
        data = gdu.sample_function(
            lambda x, y: x * (x + y),
            shape=[15, 20],
            x0=[0.5, 1.5],
            x1=[2.5, 4.5],
            ref_level=1,
        )

        hg_general = gd.HierarchicalGridData(self.grid_data_two_comp + [data])

        # Input is not a valid point
        with self.assertRaises(TypeError):
            hg_general.finest_component_at_point(0)

        # Dimensionality mismatch
        with self.assertRaises(ValueError):
            hg_general.finest_component_at_point([0])

        # Point outside the grid
        with self.assertRaises(ValueError):
            hg_general.finest_component_at_point([1000, 200])

        self.assertEqual(
            hg_general.finest_component_at_point([2, 4]), hg_general[1][0]
        )
        self.assertEqual(
            hg_general.finest_component_at_point([3, 4]), hg_general[0][0]
        )

    def test_call_evalute_with_spline(self):
        # Teting call is the same as evalute_with_spline

        hg = gd.HierarchicalGridData(self.grid_data)
        # Test with multiple components
        hg3 = gd.HierarchicalGridData(self.grid_data_two_comp)

        # Scalar input
        self.assertAlmostEqual(hg((2, 3)), 10)
        self.assertAlmostEqual(hg3((2, 3)), 10)

        # Vector input in, vector input out
        self.assertEqual(hg([(2, 3)]).shape, (1,))

        # Scalar input that pretends to be vector
        self.assertAlmostEqual(hg([(2, 3)]), 10)
        self.assertAlmostEqual(hg3([(2, 3)]), 10)

        # Vector input
        self.assertCountEqual(hg([(2, 3), (3, 2)]), [10, 12])
        self.assertCountEqual(hg3([(2, 3), (3, 2)]), [10, 12])

        def product(x, y):
            return x * (y + 2)

        # Uniform grid as input
        grid = gd.UniformGrid([3, 5], x0=[0, 1], x1=[2, 5])
        grid_data = gdu.sample_function_from_uniformgrid(product, grid)
        np.testing.assert_allclose(hg3(grid), grid_data.data)

        # Test masked
        hg_masked = km.arcsin(gd.HierarchicalGridData(self.grid_data_two_comp))

        with self.assertRaises(RuntimeError):
            hg_masked([(2, 3)])

    def test_ghost_zones_remove(self):
        hg = gd.HierarchicalGridData(self.grid_data_two_comp)

        def product(x, y):
            return x * (y + 2)

        grid_data_two_comp_no_ghost = [
            gdu.sample_function_from_uniformgrid(
                lambda x, y: x * (y + 2), g
            ).ghost_zones_removed()
            for g in self.grids1
        ]

        expected_hg = gd.HierarchicalGridData(grid_data_two_comp_no_ghost)

        hg.ghost_zones_remove()
        self.assertEqual(expected_hg, hg)

    def test_refinement_levels_merged(self):
        # This also tests to_UniformGridData

        # We redefine this to be ref_level=1
        grid1 = gd.UniformGrid([4, 5], x0=[0, 1], x1=[3, 5], ref_level=1)
        grid2 = gd.UniformGrid(
            [11, 21], x0=[4, 6], x1=[14, 26], ref_level=1, component=1
        )

        grids = [grid1, grid2]

        # Here we use the same data with another big refinement level sampled
        # from the same function
        big_grid = gd.UniformGrid(
            [16, 26], x0=[0, 1], x1=[30, 51], ref_level=0
        )
        # Big grid has resolution 2 dx of grids

        def product(x, y):
            return x * (y + 2)

        grid_data_two_comp = [
            gdu.sample_function_from_uniformgrid(product, g) for g in grids
        ]

        big_grid_data = gdu.sample_function_from_uniformgrid(product, big_grid)
        hg = gd.HierarchicalGridData(grid_data_two_comp + [big_grid_data])
        # When I merge the data I should just get big_grid at the resolution
        # of self.grid_data_two_comp
        expected_grid = gd.UniformGrid(
            [31, 51], x0=[0, 1], x1=[30, 51], ref_level=-1
        )

        expected_data = gdu.sample_function_from_uniformgrid(
            product, expected_grid
        )

        # Test with resample
        self.assertEqual(
            hg.refinement_levels_merged(resample=True), expected_data
        )

        # If we don't resample there will be points that are "wrong" because we
        # compute them with the nearest neighbors of the lowest resolution grid
        # For example, the point with coordinate (5, 1) falls inside the lowest
        # resolution grid, so its value will be the value of the closest point
        # in big_grid (6, 1) -> 18.
        self.assertEqual(hg.refinement_levels_merged()((5, 1)), 18)
        self.assertEqual(hg.refinement_levels_merged().grid, expected_grid)

        # Test a case with only one refinement level, so just returning a copy
        hg_one = gd.HierarchicalGridData([big_grid_data])
        self.assertEqual(hg_one.refinement_levels_merged(), big_grid_data)

        # Deprecated name
        with self.assertWarns(FutureWarning):
            hg.merge_refinement_levels()

    def test_coordinates(self):
        hg_coord = gd.HierarchicalGridData(self.grid_data).coordinates()
        # Test with multiple components
        hg2_coord = gd.HierarchicalGridData(
            self.grid_data_two_comp
        ).coordinates()

        self.assertAlmostEqual(hg_coord[0]((2, 3)), 2)
        self.assertAlmostEqual(hg2_coord[0]((2, 3)), 2)
        self.assertAlmostEqual(hg_coord[1]((2, 3)), 3)
        self.assertAlmostEqual(hg2_coord[1]((2, 3)), 3)

    def test_str(self):
        hg = gd.HierarchicalGridData(self.grid_data_two_comp)
        expected_str = "Available refinement levels (components):\n"
        expected_str += "0 (2)\n"
        expected_str += "Spacing at coarsest level (0): [1. 1.]\n"
        expected_str += "Spacing at finest level (0): [1. 1.]"
        self.assertEqual(expected_str, hg.__str__())

    def test_partial_differentiated(self):
        # Here we are also testing _call_component_method

        geom = gd.UniformGrid(
            [8001, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=0
        )
        geom2 = gd.UniformGrid(
            [10001, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=1
        )

        sin_wave1 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom
        )
        sin_wave2 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom2
        )
        original_sin1 = sin_wave1.copy()
        original_sin2 = sin_wave2.copy()

        sin_wave = gd.HierarchicalGridData([sin_wave1] + [sin_wave2])
        sin_copy = sin_wave.copy()

        # Second derivative should still be a -sin
        sin_wave.partial_differentiate(0, order=2)

        np.testing.assert_allclose(
            -sin_wave[0][0].data, original_sin1.data, atol=1e-3
        )
        np.testing.assert_allclose(
            -sin_wave[1][0].data, original_sin2.data, atol=1e-3
        )

        # Test _call_component_method with non-string name
        with self.assertRaises(TypeError):
            sin_wave._call_component_method(sin_wave)

        # Test _call_component_method with non existing method
        with self.assertRaises(ValueError):
            sin_wave._call_component_method("lol")

        gradient = sin_copy.gradient(order=2)
        # Along the first direction (it's a HierarchicalGridData)
        partial_x = gradient[0]

        np.testing.assert_allclose(
            -partial_x[0][0].data, original_sin1.data, atol=1e-3
        )

        # First refinement_level
        np.testing.assert_allclose(
            -partial_x[1][0].data, original_sin2.data, atol=1e-3
        )

    def test_slice(self):
        hg = gd.HierarchicalGridData(self.grid_data)

        # Test cut is outside grid
        with self.assertRaises(ValueError):
            hg.slice([-100, -200])

        # Test other error (captured matching the error message)
        with self.assertRaises(ValueError):
            hg.slice([1, 2, 3])

        # hg splits the rectangle with x0 = [0, 1], x1 = [14, 26] in four pieces
        # (self.grid0). If we cut at x = 2, we should only get two pieces,
        # patch1 and patch4, the first and the last elements of self.grid_data

        cut = [2, None]
        expected_hg = gd.HierarchicalGridData(
            [self.grid_data[0].sliced(cut), self.grid_data[3].sliced(cut)]
        )

        hg.slice(cut)

        self.assertEqual(hg, expected_hg)

    def test_coordinates_at(self):
        # Here we are also testing _call_component_method

        geom = gd.UniformGrid(
            [8001, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=0
        )
        geom2 = gd.UniformGrid(
            [10001, 3], x0=[0, 0], x1=[2 * np.pi, 1], ref_level=1
        )

        sin_wave1 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom
        )

        sin_wave2 = gdu.sample_function_from_uniformgrid(
            lambda x, y: np.sin(x), geom2
        )

        sin_wave = gd.HierarchicalGridData([sin_wave1] + [sin_wave2])

        # We are taking the abs
        self.assertEqual(sin_wave.coordinates_at_minimum()[0], 0)

        point = sin_wave.coordinates_at_maximum()

        np.testing.assert_allclose(sin_wave(point), sin_wave.abs_max())

        point_min = sin_wave.coordinates_at_minimum(absolute=False)

        np.testing.assert_allclose(sin_wave(point_min), sin_wave.min())
