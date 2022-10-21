#!/usr/bin/env python3

# Copyright (C) 2022 Gabriele Bozzola
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

from kuibit import tensor as kvm
from kuibit.frequencyseries import FrequencySeries
from kuibit.grid_data import UniformGridData
from kuibit.grid_data_utils import sample_function
from kuibit.timeseries import TimeSeries


class TestTensor(unittest.TestCase):
    def setUp(self):

        self.times = np.linspace(np.pi / 3, np.pi / 2, 100)
        self.values = np.sin(self.times)
        self.ts = TimeSeries(self.times, self.values)

        self.ugd = sample_function(
            lambda x: np.sin(x),
            x0=[self.times[0]],
            x1=[self.times[-1]],
            shape=len(self.times),
        )

        self.test_data = {TimeSeries: self.ts, UniformGridData: self.ugd}

    def test_init(self):

        # Data is not iterable
        with self.assertRaises(TypeError):
            kvm.Tensor(1)

        # Data is empty
        with self.assertRaises(TypeError):
            kvm.Tensor([])

        # Data is not derived from BaseNumerical
        with self.assertRaises(TypeError):
            kvm.Tensor([1])

        # Test data with incosistent shape
        data = [[self.ts], [self.ts, self.ts]]

        with self.assertRaises(RuntimeError):
            kvm.Tensor(data)

        # Test data with incosistent depth
        with self.assertRaises(RuntimeError):
            kvm.Tensor([[self.ts], [[self.ts]]])

        # Data is inconsistent type
        with self.assertRaises(TypeError):
            kvm.Tensor([self.ts, self.ugd])

        # Test that we traverse the tree in the expected order
        # The (1, 2, 3) matrix: [[[a,b, c], [d, d, f]]]
        # should be represented in order

        tss = [TimeSeries(self.times, i * self.values) for i in range(6)]

        matrix = [[[tss[0], tss[1], tss[2]], [tss[3], tss[4], tss[5]]]]

        mat_Ten = kvm.Tensor(matrix)

        self.assertCountEqual(mat_Ten.flat_data, tss)
        self.assertCountEqual(mat_Ten.shape, (1, 2, 3))

    def test_type(self):
        self.assertEqual(kvm.Tensor([self.ts]).type, TimeSeries)

    def test_shape(self):
        self.assertCountEqual(kvm.Tensor([self.ts]).shape, (1,))

    def test_from_shape_and_flat_data(self):

        tss = [TimeSeries(self.times, i * self.values) for i in range(6)]
        matrix = [[[tss[0], tss[1], tss[2]], [tss[3], tss[4], tss[5]]]]
        mat_Ten = kvm.Tensor(matrix)

        mat_Ten2 = kvm.Tensor.from_shape_and_flat_data((1, 2, 3), tss)

        self.assertCountEqual(mat_Ten2.flat_data, mat_Ten.flat_data)
        self.assertCountEqual(mat_Ten2.shape, mat_Ten.shape)

    def test_data(self):

        tss = [TimeSeries(self.times, i * self.values) for i in range(6)]
        matrix = [[[tss[0], tss[1], tss[2]], [tss[3], tss[4], tss[5]]]]
        mat_Ten = kvm.Tensor(matrix)

        self.assertCountEqual(mat_Ten.data, matrix)

    def test_eq(self):

        for type_, data in self.test_data.items():
            with self.subTest(type_=type_):
                vec = kvm.Tensor([data])
                mat = kvm.Tensor([[data, data], [data, data]])

                self.assertEqual(vec, vec)

                # Three ways to fail
                self.assertNotEqual(vec, 2)
                self.assertNotEqual(vec, mat)
                self.assertNotEqual(vec, kvm.Tensor([2 * data]))

    def test_call(self):
        # Series
        mat = kvm.Tensor([[self.ts, self.ts], [self.ts, self.ts]])
        expe = self.ts(1.25)
        np.testing.assert_allclose(
            mat(1.25), np.array([[expe, expe], [expe, expe]])
        )

        # Grid Data
        mat = kvm.Tensor([[self.ugd, self.ugd], [self.ugd, self.ugd]])
        expe = self.ugd([1.25])
        np.testing.assert_allclose(
            mat([1.25]), np.array([[expe, expe], [expe, expe]])
        )

    def test_unary_functions(self):

        tens = {
            TimeSeries: kvm.Tensor([self.ts]),
            UniformGridData: kvm.Tensor([self.ugd]),
        }

        def test_f(fun, type_):
            out = fun(tens[type_])
            expected_out = kvm.Tensor([fun(self.test_data[type_])])
            self.assertEqual(out, expected_out)

        for fun in [
            np.abs,
            np.sin,
            np.cos,
            np.tan,
            np.arcsin,
            np.arccos,
            np.sinh,
            np.cosh,
            np.tanh,
            np.arcsinh,
            np.sqrt,
            np.exp,
            np.log,
            np.log2,
            np.log10,
            np.conj,
            np.conjugate,
        ]:
            with self.subTest(fun=fun):
                for type_ in self.test_data:
                    with self.subTest(type_=type_):
                        test_f(fun, type_)

        times2 = np.linspace(0.25, 0.5, 100)
        values2 = np.array([1.5] * len(times2))
        ts2 = TimeSeries(times2, values2)

        ugd2 = [
            sample_function(
                lambda x: 1.5,
                x0=[times2[0]],
                x1=[times2[1]],
                shape=len(times2),
            )
            for i in range(1, 3)
        ]

        test_data2 = {TimeSeries: ts2, UniformGridData: ugd2}

        tens2 = {
            TimeSeries: kvm.Tensor([ts2]),
            UniformGridData: kvm.Tensor([ugd2]),
        }

        def test_f2(fun, type_):
            out = fun(tens2[type_])
            expected_out = kvm.Tensor([fun(test_data2[type_])])
            self.assertEqual(out, expected_out)

        # Different domain
        for fun in [np.arccosh]:
            with self.subTest(fun=fun):
                for type_ in test_data2:
                    with self.subTest(type_=type_):
                        test_f2(fun, type_)

        times3 = np.linspace(0.25, 0.5, 100)
        values3 = np.array([0.75] * len(times3))
        ts3 = TimeSeries(times3, values3)

        ugd3 = [
            sample_function(
                lambda x: 0.75,
                x0=[times3[0]],
                x1=[times3[1]],
                shape=len(times3),
            )
            for i in range(1, 3)
        ]

        test_data3 = {TimeSeries: ts3, UniformGridData: ugd3}

        tens3 = {
            TimeSeries: kvm.Tensor([ts3]),
            UniformGridData: kvm.Tensor([ugd3]),
        }

        def test_f3(fun, type_):
            out = fun(tens3[type_])
            expected_out = kvm.Tensor([fun(test_data3[type_])])
            self.assertEqual(out, expected_out)

        # Different domain
        for fun in [np.arctanh]:
            with self.subTest(fun=fun):
                for type_ in test_data3:
                    with self.subTest(type_=type_):
                        test_f3(fun, type_)

    def test_copy(self):

        for type_, data in self.test_data.items():
            with self.subTest(type_=type_):
                mat = kvm.Tensor([[data, data], [data, data]])
                mat_copy = mat.copy()

                self.assertEqual(mat, mat_copy)
                self.assertIsNot(mat, mat_copy)

    def test_reduction(self):

        # We only test one reduction
        for type_, data in self.test_data.items():
            with self.subTest(type_=type_):
                mat = kvm.Tensor([[data, data], [data, data]])
                mean = data.mean()
                np.testing.assert_allclose(
                    mat.mean(), np.array([[mean, mean], [mean, mean]])
                )

    def test_apply_to_self(self):

        # We only test one reduction
        for type_, data in self.test_data.items():
            with self.subTest(type_=type_):
                mat = kvm.Tensor([[data, data], [data, data]])
                mat_copy = mat.copy()
                mat._apply_to_self(lambda x: 2 * x)
                self.assertEqual(mat, 2 * mat_copy)

    def test_getattr(self):

        # We test with series
        mat = kvm.Tensor([[self.ts, self.ts], [self.ts, self.ts]])

        # No attribute
        with self.assertRaises(AttributeError):
            mat.lol()

        # Returning a property
        dt = self.ts.dt
        np.testing.assert_allclose(mat.dt, np.array([[dt, dt], [dt, dt]]))

        # Returning a number
        t_at_max = self.ts.time_at_maximum()
        np.testing.assert_allclose(
            mat.time_at_maximum(),
            np.array([[t_at_max, t_at_max], [t_at_max, t_at_max]]),
        )

        # Returning a TimeSeries
        ts_mean_removed = self.ts.mean_removed()
        self.assertEqual(
            mat.mean_removed(),
            kvm.Tensor(
                [
                    [ts_mean_removed, ts_mean_removed],
                    [ts_mean_removed, ts_mean_removed],
                ]
            ),
        )

        # Returning a FrequencySeries
        with self.assertRaises(RuntimeError):
            mat.to_FrequencySeries()

        # Editing in place
        ts_copy = self.ts.copy()
        mat_mean = kvm.Tensor([ts_copy])
        mat_mean.mean_remove()
        self.assertEqual(mat_mean, kvm.Tensor([ts_mean_removed]))

    def test_apply_binary(self):

        # We only test one operation
        for type_, data in self.test_data.items():
            with self.subTest(type_=type_):
                mat = kvm.Tensor([[data, data], [data, data]])

                # Test adding a number
                exp_data = data + 2
                exp_out = kvm.Tensor(
                    [[exp_data, exp_data], [exp_data, exp_data]]
                )

                self.assertEqual(mat + 2, exp_out)

                # Test adding an object of the same type
                exp_data = data + data
                exp_out = kvm.Tensor(
                    [[exp_data, exp_data], [exp_data, exp_data]]
                )

                self.assertEqual(mat + data, exp_out)

                # Test adding a tensor with correct shape
                self.assertEqual(mat + mat, 2 * mat)

                # Test adding a tensor with wrong shape
                with self.assertRaises(ValueError):
                    mat + kvm.Tensor([data])

                # Test adding an object of the wrong type
                with self.assertRaises(ValueError):
                    data_freq = FrequencySeries([1], [2])
                    mat + kvm.Tensor(
                        [[data_freq, data_freq], [data_freq, data_freq]]
                    )

    def test_matmul(self):
        ten = kvm.Tensor([self.ts])
        with self.assertRaises(TypeError):
            1 @ ten
        with self.assertRaises(TypeError):
            ten @ 1


class TestVector(unittest.TestCase):
    def setUp(self):
        times = np.linspace(np.pi / 3, np.pi / 2, 100)
        self.ts = [TimeSeries(times, np.sin(i * times)) for i in range(1, 4)]

        self.vec = kvm.Vector(self.ts)

    def test_len(self):
        self.assertEqual(len(self.vec), len(self.ts))

    def test_getitem(self):
        self.assertEqual(self.vec[1], self.ts[1])

    def test_dot(self):

        # Not a vector
        with self.assertRaises(TypeError):
            self.vec.dot(1)

        # Incompatible shapes
        with self.assertRaises(RuntimeError):
            vec2 = kvm.Vector([self.ts[0], self.ts[1]])
            self.vec.dot(vec2)

        # Incompatible types
        with self.assertRaises(TypeError):
            freq = FrequencySeries([0], [1])
            vec3 = kvm.Vector([freq] * len(self.ts))
            self.vec.dot(vec3)

        times = np.linspace(np.pi / 3, np.pi / 2, 100)

        ts1 = [TimeSeries(times, np.sin(i * times)) for i in range(1, 3)]
        ts2 = [TimeSeries(times, np.cos(i * times)) for i in range(1, 3)]

        vec1 = kvm.Vector(ts1)

        vec2 = kvm.Vector(ts2)

        dot = vec1.dot(vec2)
        expected = sum(t1 * t2 for t1, t2 in zip(ts1, ts2))

        self.assertEqual(dot, expected)

    def test_norm(self):
        self.assertEqual(
            self.vec.norm(), (sum(t**2 for t in self.ts)).sqrt()
        )


class TestMatrix(unittest.TestCase):
    def setUp(self):
        times = np.linspace(np.pi / 3, np.pi / 2, 100)
        self.ts = [
            [TimeSeries(times, np.sin(i * times)) for i in range(1, 4)],
            [TimeSeries(times, np.sin(2 * i * times)) for i in range(1, 4)],
        ]

        self.mat = kvm.Matrix(self.ts)

    def test_getitem(self):
        self.assertIs(self.mat[1, 2], self.ts[1][2])
