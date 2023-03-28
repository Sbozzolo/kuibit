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

"""Tests for kuibit.timeseries
"""

import os
import unittest
from unittest import mock

import numpy as np
from scipy import signal

from kuibit import numerical, series
from kuibit import timeseries as ts


class TestTimeseries(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0, 2 * np.pi, 100)
        self.values = np.sin(self.times)

        self.TS = ts.TimeSeries(self.times, self.values)
        # Complex
        self.TS_c = ts.TimeSeries(self.times, self.values + 1j * self.values)

        # Complex and masked
        self.vals_cm = np.ma.log10(self.values + 1j * self.values)
        self.TS_cm = ts.TimeSeries(self.times, self.vals_cm)

    # This is to make coverage happy and test the abstract methods
    # There's no real test here
    @mock.patch.multiple(numerical.BaseNumerical, __abstractmethods__=set())
    def test_numerical(self):
        abs_numerical = numerical.BaseNumerical()
        with self.assertRaises(NotImplementedError):
            abs_numerical._apply_unary(lambda x: x)
        with self.assertRaises(NotImplementedError):
            abs_numerical._apply_binary(0, lambda x: x)
        with self.assertRaises(NotImplementedError):
            abs_numerical._apply_reduction(lambda x: x)
        with self.assertRaises(NotImplementedError):
            abs_numerical._apply_to_self(lambda x: x)

    def test__make_array(self):
        # A number
        self.assertTrue(isinstance(self.TS._make_array(1), np.ndarray))
        # A list
        self.assertTrue(isinstance(self.TS._make_array([1, 2]), np.ndarray))
        # An array
        arr = np.array([1, 2])
        self.assertTrue(isinstance(self.TS._make_array(arr), np.ndarray))

    def test___return_array_if_monotonic(self):
        # The time is not monotonically increasing
        times = np.linspace(2 * np.pi, 0, 100)

        with self.assertRaises(ValueError):
            self.TS._return_array_if_monotonic(times)

        # A number
        self.assertCountEqual(
            self.TS._return_array_if_monotonic(np.array([1])), np.array([1])
        )
        # A list
        self.assertCountEqual(
            self.TS._return_array_if_monotonic(np.array([1, 2])),
            np.array([1, 2]),
        )

    def test_init(self):
        # Check that errors are thrown if:
        # 1. There is a mismatch between t and y
        # 2. The timeseries is empty

        # 1
        times = np.linspace(0, 2 * np.pi, 100)
        values = np.array([1, 2, 3])

        with self.assertRaises(ValueError):
            ts.TimeSeries(times, values)

        # 2
        times = np.array([])
        values = np.array([])

        with self.assertRaises(ValueError):
            ts.TimeSeries(times, values)

        # Test timeseries with one element
        scalar = ts.TimeSeries(0, 0)

        self.assertEqual(scalar.y, 0)

        # Test guarantee_x_is_monotonic
        # This should not throw an error even if it is wrong
        times = np.linspace(2 * np.pi, 0, 100)
        values = np.sin(times)
        wrong = ts.TimeSeries(times, values, guarantee_t_is_monotonic=True)
        self.assertCountEqual(wrong.t, times)

    def test_setters(self):
        # Cannot change the array length
        with self.assertRaises(ValueError):
            self.TS.t = [1]
        with self.assertRaises(ValueError):
            self.TS.y = [1]

        # Cannot have a non-monotonic array
        with self.assertRaises(ValueError):
            self.TS.t = np.linspace(2 * np.pi, 0, 100)

        # Set with valid data
        self.TS.t = self.TS.t
        # Check that the spline is invalid
        self.assertTrue(self.TS.invalid_spline)

        # Test getter for plotting
        self.assertIs(self.TS.index.values, self.TS.x)
        self.assertIs(self.TS.values, self.TS.y)

    def test_to_numpy(self):
        # to_numpy is used by matplotlib >= 3.5.2 for implicit plotting
        self.assertIs(self.TS.to_numpy(), self.TS.y)
        self.assertIs(self.TS.index.to_numpy(), self.TS.x)

    def test_len(self):
        self.assertEqual(len(self.TS), 100)

        self.assertEqual(len(self.TS_cm), 100)

    def test_iter(self):
        for t, y in self.TS:
            self.assertEqual((t, y), (self.TS.t[0], self.TS.y[0]))
            break

        # Masked
        for t, y in self.TS_cm:
            self.assertEqual((t, y), (self.TS_cm.t[0], self.TS_cm.y[0]))
            break

    def test_x_at_maximum_minimum_y(self):
        t = np.linspace(0, 1, 100)

        self.assertEqual(ts.TimeSeries(t, t).time_at_maximum(), 1)
        self.assertEqual(ts.TimeSeries(t, t + 1j * t).time_at_maximum(), 1)
        self.assertEqual(ts.TimeSeries(t, t + 1j * t).time_at_minimum(), 0)

        # Passing the absolute argument
        self.assertEqual(
            ts.TimeSeries(t, -t).time_at_maximum(absolute=False), 0
        )
        self.assertEqual(
            ts.TimeSeries(t, -t).time_at_minimum(absolute=False), 1
        )

        # Masked

        # 0 should be one the points
        t = np.linspace(-1, 1, 99)

        self.assertAlmostEqual(
            ts.TimeSeries(
                t, np.ma.masked_greater_equal(t, 0)
            ).time_at_minimum(),
            0,
        )

    def test_align_maximum_minimum(self):
        t = np.linspace(0, 1, 100)

        tseries = ts.TimeSeries(t, t)

        # Should shift everything of -1, so new times should be from -1 to 0
        tseries.align_at_maximum()

        np.testing.assert_allclose(tseries.t, t - 1)

        # Minimum is at t = 1, notice we use the above t, so this is a line
        # that goes from -1 to 0. The absolute minimum is at the end.
        y2 = np.linspace(-1, 0, 100)
        tseries2 = ts.TimeSeries(t, y2)

        tseries2.align_at_minimum()

        np.testing.assert_allclose(tseries2.t, t - 1)

    def test_is_regularly_sampled(self):
        self.assertTrue(self.TS.is_regularly_sampled())

        # Log space is not regular
        times = np.logspace(0, 1, 100)
        ts_log = ts.TimeSeries(times, self.values)
        self.assertFalse(ts_log.is_regularly_sampled())

        # If the series is only one point long, an error should be raised
        with self.assertRaises(RuntimeError):
            ts.TimeSeries([1], [1]).is_regularly_sampled()

    def test_tmin_tmax_length_dt(self):
        # Testing methods of the base class
        self.assertAlmostEqual(self.TS.xmin, 0)
        self.assertAlmostEqual(self.TS.xmax, 2 * np.pi)

        self.assertAlmostEqual(self.TS.tmin, 0)
        self.assertAlmostEqual(self.TS.tmax, 2 * np.pi)

        times = np.linspace(1, 2, 100)

        sins = ts.TimeSeries(times, self.values)

        self.assertAlmostEqual(sins.time_length, 1)
        self.assertAlmostEqual(sins.duration, 1)

        self.assertAlmostEqual(self.TS.dt, self.TS.tmax / (len(self.TS) - 1))

        with self.assertRaises(ValueError):
            sins.t[-1] = 20
            sins.dt

    def test__apply_binary(self):
        # Check that errors are thrown if:
        # 1. Lists have different times
        # 2. The object is unknown

        # We as example of a function np.sum

        # 1a
        times = np.linspace(0, 2 * np.pi, 200)
        values = np.cos(times)
        cos = ts.TimeSeries(times, values)

        with self.assertRaises(ValueError):
            _ = self.TS._apply_binary(cos, np.add)

        # 1b
        times = np.linspace(2, 3 * np.pi, 100)
        values = np.cos(times)
        cos = ts.TimeSeries(times, values)

        with self.assertRaises(ValueError):
            _ = self.TS._apply_binary(cos, np.add)

        # 2
        with self.assertRaises(TypeError):
            self.TS._apply_binary("str", np.add)

    def test_add(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        out = self.TS + ts.TimeSeries(times, values)
        np.testing.assert_allclose(out.y, 2 * np.sin(times))

        # Scalar
        out = self.TS + 1
        np.testing.assert_allclose(out.y, 1 + np.sin(times))

        out = 1 + self.TS
        np.testing.assert_allclose(out.y, 1 + np.sin(times))

        # Test iadd
        out += 1
        np.testing.assert_allclose(out.y, 2 + np.sin(times))

        out += self.TS
        np.testing.assert_allclose(out.y, 2 + 2 * np.sin(times))

        # Everything should work with masked data too
        out = self.TS_cm + ts.TimeSeries(times, values)

        np.testing.assert_allclose(out.y, self.vals_cm + values)

        # Scalar
        out = self.TS_cm + 1
        np.testing.assert_allclose(out.y, 1 + self.vals_cm)

        out = 1 + self.TS_cm
        np.testing.assert_allclose(out.y, 1 + self.vals_cm)

        # Test iadd
        out += 1
        np.testing.assert_allclose(out.y, 2 + self.vals_cm)

        out += self.TS_cm
        np.testing.assert_allclose(out.y, 2 + 2 * self.vals_cm)

    def test_sub(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        out = self.TS - ts.TimeSeries(times, values)
        np.testing.assert_allclose(out.y, 0)

        # Scalar
        out = self.TS - 1
        np.testing.assert_allclose(out.y, np.sin(times) - 1)

        out = 1 - self.TS
        np.testing.assert_allclose(out.y, 1 - np.sin(times), atol=1e-14)

        # Test isub
        out -= 1
        np.testing.assert_allclose(out.y, -np.sin(times), atol=1e-14)

        out -= self.TS
        np.testing.assert_allclose(out.y, -2 * np.sin(times), atol=1e-14)

    def test_neg(self):
        np.testing.assert_allclose((-self.TS).y, -np.sin(self.times))

    def test_abs(self):
        np.testing.assert_allclose(abs(self.TS).y, np.abs(np.sin(self.times)))
        np.testing.assert_allclose(
            np.abs(self.TS).y, np.abs(np.sin(self.times))
        )
        np.testing.assert_allclose(self.TS.abs().y, np.abs(np.sin(self.times)))

        # Masked
        np.testing.assert_allclose(self.TS_cm.abs().y, np.abs(self.vals_cm))

    def test_mul(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        out = self.TS * ts.TimeSeries(times, values)
        np.testing.assert_allclose(out.y, np.sin(times) ** 2)

        # Scalar
        out = self.TS * 3
        np.testing.assert_allclose(out.y, np.sin(times) * 3)

        out = 3 * self.TS
        np.testing.assert_allclose(out.y, np.sin(times) * 3)

        # Test imul
        out *= 3
        np.testing.assert_allclose(out.y, 9 * np.sin(times))

        out *= self.TS
        np.testing.assert_allclose(out.y, 9 * np.sin(times) ** 2)

    def test_div(self):
        # Errors are tested by test_apply_binary

        with self.assertRaises(ValueError):
            out = self.TS / 0

        # Let's avoid division by 0

        times = np.linspace(np.pi / 3, np.pi / 2, 100)
        values = np.sin(times)

        sins = ts.TimeSeries(times, values)

        out = sins / sins
        np.testing.assert_allclose(out.y, 1)

        # Test rtrudiv
        out = 1 / sins
        np.testing.assert_allclose(out.y, 1 / values)

        # Scalar
        out = sins / 3
        np.testing.assert_allclose(out.y, np.sin(times) / 3)

        # Test itruediv
        out /= 2
        np.testing.assert_allclose(out.y, np.sin(times) / 6)

        out /= sins
        np.testing.assert_allclose(out.y, 1 / 6)

    def test_power(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.array([2] * len(times))

        out = self.TS ** ts.TimeSeries(times, values)
        np.testing.assert_allclose(out.y, np.sin(times) ** 2)

        # Scalar
        out = self.TS**2
        np.testing.assert_allclose(out.y, np.sin(times) ** 2)

        # Test ipow
        out **= 2
        np.testing.assert_allclose(out.y, np.sin(times) ** 4)

        out **= ts.TimeSeries(times, values)
        np.testing.assert_allclose(out.y, np.sin(times) ** 8)

    def test_eq(self):
        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        sins = ts.TimeSeries(times, values)

        self.assertFalse(sins == 1)
        self.assertTrue(self.TS == sins)

    def test_is_complex(self):
        self.assertFalse(self.TS.is_complex())
        self.assertTrue(self.TS_c.is_complex())

    def test_is_masked(self):
        self.assertFalse(self.TS.is_masked())
        self.assertTrue(self.TS_cm.is_masked())

    def test_unary_functions(self):
        def test_f(f):
            times = np.linspace(np.pi / 3, np.pi / 2, 100)
            values = np.array([0.5] * len(times))

            out = f(ts.TimeSeries(times, values))
            np.testing.assert_allclose(out.y, f(values))

        def test_f2(f):
            times = np.linspace(np.pi / 3, np.pi / 2, 100)
            values = np.array([1.5] * len(times))

            out = f(ts.TimeSeries(times, values))
            np.testing.assert_allclose(out.y, f(values))

        for f in [
            np.abs,
            np.sin,
            np.cos,
            np.tan,
            np.arcsin,
            np.arccos,
            np.arctan,
            np.sinh,
            np.cosh,
            np.tanh,
            np.arcsinh,
            np.arctanh,
            np.sqrt,
            np.exp,
            np.log,
            np.log2,
            np.log10,
            np.conj,
            np.conjugate,
        ]:
            with self.subTest(f=f):
                test_f(f)

        # Different domain
        for f in [np.arccosh]:
            with self.subTest(f=f):
                test_f2(f)

        # Different call method
        out = (self.TS).real()
        np.testing.assert_allclose(out.y, np.sin(self.times))
        out = (self.TS).imag()
        np.testing.assert_allclose(out.y, 0)

    def test_min_max(self):
        self.assertEqual(self.TS.min(), np.min(self.TS.y))
        self.assertEqual(self.TS.max(), np.max(self.TS.y))
        self.assertEqual(self.TS.abs_min(), np.min(np.abs(self.TS.y)))
        self.assertEqual(self.TS.abs_max(), np.max(np.abs(self.TS.y)))

        # Add a NaN and test the nan-methods
        TS_c = self.TS.copy()

        TS_c.y[0] = np.nan

        self.assertEqual(self.TS.nanmin(), np.nanmin(self.TS.y))
        self.assertEqual(self.TS.nanmax(), np.nanmax(self.TS.y))
        self.assertEqual(self.TS.abs_nanmin(), np.nanmin(np.abs(self.TS.y)))
        self.assertEqual(self.TS.abs_nanmax(), np.nanmax(np.abs(self.TS.y)))

    def test_mean_median_std(self):
        self.assertEqual(self.TS.mean(), np.mean(self.TS.y))
        self.assertEqual(self.TS.average(), np.mean(self.TS.y))
        self.assertEqual(self.TS.median(), np.median(self.TS.y))
        self.assertEqual(self.TS.std(), np.std(self.TS.y))

    def test_zero_pad(self):
        # Check invalid number of points

        with self.assertRaises(ValueError):
            self.TS.zero_pad(50)

        # Check unevenly-space time
        times = np.logspace(0, 1, 100)
        sins = ts.TimeSeries(times, self.values)

        with self.assertRaises(ValueError):
            sins.zero_pad(200)

        sins2 = ts.TimeSeries(self.times, self.values)

        sins2.zero_pad(200)

        np.testing.assert_allclose(sins2.y[100:], np.zeros(100)), 1

        # Check if the sequence is still equispaced in time
        dts = np.diff(sins2.t)
        dt0 = dts[0]
        np.testing.assert_allclose(dts, dt0)

    def test_mean_remove(self):
        # Check unevenly-space time
        times = np.logspace(0, 1, 100)
        sins = ts.TimeSeries(times, self.values + 1)

        np.testing.assert_allclose(
            sins.mean_removed().y, self.values, atol=1e-14
        )

        sins.mean_remove()

        np.testing.assert_allclose(sins.y, self.values, atol=1e-14)

    def test_initial_final_time_remove(self):
        # Check unevenly-space time
        times = np.linspace(-1, 3, 100)
        sins = ts.TimeSeries(times, self.values + 1)

        # Remove the first 1
        sins.initial_time_remove(1)

        new_times = times[times > 0]
        np.testing.assert_allclose(sins.t, new_times)

        # Remove the last 1
        sins.final_time_remove(1)

        new_times = new_times[new_times < 2]
        np.testing.assert_allclose(sins.t, new_times)

    def test_copy(self):
        tscopy = self.TS.copy()
        tscopyc = self.TS_c.copy()
        self.assertEqual(self.TS, tscopy)
        self.assertEqual(self.TS_c, tscopyc)
        self.assertEqual(self.TS_c.invalid_spline, tscopyc.invalid_spline)

        # Test copy also with splines
        tscopyc._make_spline()
        tscopyc2 = tscopyc.copy()
        self.assertEqual(tscopyc.spline_imag, tscopyc2.spline_imag)

    def test_time_shift(self):
        times = np.logspace(0, 1, 100)
        sins = ts.TimeSeries(times, self.values)

        np.testing.assert_allclose(sins.time_shifted(1).t, times + 1)

        sins.time_shift(1)

        np.testing.assert_allclose(sins.t, times + 1)

    def test_phase_shift(self):
        sins = ts.TimeSeries(self.times, self.values)

        np.testing.assert_allclose(
            sins.phase_shifted(np.pi / 2).y, 1j * self.values
        )
        sins.phase_shift(np.pi / 2)

        np.testing.assert_allclose(sins.y, 1j * self.values)

    def test_crop(self):
        sins = ts.TimeSeries(self.times, self.values)

        self.assertGreaterEqual(sins.cropped(init=1).tmin, 1)
        self.assertLessEqual(sins.cropped(end=1).tmax, 1)
        self.assertGreaterEqual(sins.clipped(init=1).tmin, 1)
        self.assertLessEqual(sins.clipped(end=1).tmin, 1)

        sins.crop(init=0.5, end=1.5)

        self.assertGreaterEqual(sins.tmin, 0.5)
        self.assertLessEqual(sins.tmax, 1.5)

        sins.clip(init=1, end=1.4)
        self.assertGreaterEqual(sins.tmin, 1)
        self.assertLessEqual(sins.tmax, 1.4)

    def test_local_minima_maxima(self):
        times = np.linspace(0, 2 * np.pi, 5000)
        values = np.cos(times)

        coss = ts.TimeSeries(times, values)

        # Maxima are in 0, pi
        maxima_x, maxima_y = coss.local_maxima()

        np.testing.assert_allclose(maxima_x, [0, 2 * np.pi], atol=5e-4)
        np.testing.assert_allclose(maxima_y, [1, 1], atol=5e-4)

        # Test minima with a complex signal, it should take the absolute

        values = np.cos(times) + 1j * np.cos(times)
        complex_coss = ts.TimeSeries(times, values)

        # Minima are in pi/2, 3 * pi /2
        minima_x, minima_y = complex_coss.local_minima()
        np.testing.assert_allclose(
            minima_x, [np.pi / 2, 3 * np.pi / 2], atol=5e-4
        )
        np.testing.assert_allclose(minima_y, [0, 0], atol=5e-4)

        # Test with a contrived example
        xs = [0, 2, 3]
        ys = [0, 1, 0]

        maxima_x2, maxima_y2 = ts.TimeSeries(xs, ys).local_maxima()
        self.assertEqual(maxima_x2[0], 2)
        self.assertEqual(maxima_y2[0], 1)

    def test_save(self):
        times = np.logspace(0, 1, 10)
        sins = ts.TimeSeries(times, np.sin(times))
        compl = ts.TimeSeries(times, np.sin(times) + 1j * np.sin(times))

        sins_file = "test_save_sins.dat"
        compl_file = "test_save_compl.dat"

        sins.save(sins_file)
        compl.save(compl_file)

        loaded_sins = np.loadtxt(sins_file).T
        os.remove(sins_file)

        loaded_compl = np.loadtxt(compl_file).T
        os.remove(compl_file)

        np.testing.assert_allclose(loaded_sins[0], times)
        np.testing.assert_allclose(loaded_sins[1], np.sin(times))

        np.testing.assert_allclose(loaded_compl[0], times)
        np.testing.assert_allclose(loaded_compl[1], np.sin(times))
        np.testing.assert_allclose(loaded_compl[2], np.sin(times))

        with self.assertWarns(RuntimeWarning):
            path = "/tmp/tmp_kuibit_tmp.dat"
            self.TS_cm.save("/tmp/tmp_kuibit_tmp.dat")
            os.remove(path)

    def test_nans_remove(self):
        values = self.values[:]
        values[0] = np.inf
        values[-1] = np.nan
        sins = ts.TimeSeries(self.times, values)

        np.testing.assert_allclose(sins.nans_removed().y, values[1:-1])
        np.testing.assert_allclose(sins.nans_removed().t, self.times[1:-1])

        sins.nans_remove()

        np.testing.assert_allclose(sins.y, values[1:-1])
        np.testing.assert_allclose(sins.t, self.times[1:-1])

    def test_make_spline_call(self):
        # Cannot make a spline with 1 point
        with self.assertRaises(ValueError):
            sins = ts.TimeSeries([0], [0])
            sins._make_spline()

        # Check that spline reproduce data
        # These are pulled from the data
        np.testing.assert_allclose(self.TS(self.times), self.values)
        # These have some that computed with splines
        other_times = np.linspace(0, np.pi, 100)
        other_values = np.sin(other_times)

        np.testing.assert_allclose(
            self.TS(other_times), other_values, atol=1e-7
        )

        np.testing.assert_allclose(self.TS(np.pi / 2), 1)

        # Vector input in, vector input out
        self.assertTrue(isinstance(self.TS([np.pi / 2]), np.ndarray))

        np.testing.assert_allclose(self.TS(self.TS.t[0]), self.TS.y[0])

        # From data
        np.testing.assert_allclose(
            self.TS_c(self.times), self.values + 1j * self.values
        )

        # From spline
        np.testing.assert_allclose(
            self.TS_c(other_times), other_values + 1j * other_values, atol=1e-7
        )

        # Masked data
        with self.assertRaises(RuntimeError):
            self.TS_cm(other_times)

        # Does the spline update?
        # Let's test with a method that changes the timeseries
        sins = ts.TimeSeries(self.times, self.values + 1)
        np.testing.assert_allclose(sins(self.times), self.values + 1)

        sins.mean_remove()

        np.testing.assert_allclose(sins(self.times), self.values, atol=1e-14)

        with self.assertRaises(ValueError):
            sins(-1)

        # Test that setting directly the members invalidates the spline
        sins.y *= 2
        self.assertTrue(sins.invalid_spline)
        sins.invalid_spline = False
        sins.t *= 2
        self.assertTrue(sins.invalid_spline)

    def test_resample(self):
        new_times = np.array([float(1e-5 * i**2) for i in range(0, 100)])

        # Test no resampling
        sins = self.TS.copy()
        np.testing.assert_allclose(sins.resampled(sins.t).y, sins.y)
        np.testing.assert_allclose(sins.resampled(sins.t).t, sins.t)

        np.testing.assert_allclose(
            sins.resampled(new_times).y,
            np.sin(new_times),
            atol=1e-7,
            rtol=3e-6,
        )
        np.testing.assert_allclose(sins.resampled(new_times).t, new_times)

        sins.resample(new_times)

        np.testing.assert_allclose(
            sins.y, np.sin(new_times), atol=1e-7, rtol=3e-6
        )
        np.testing.assert_allclose(sins.t, new_times)

        # Test regular_sample using sins, that now is unevenly
        # sampled
        regular_times = np.linspace(0, new_times[-1], 100)

        np.testing.assert_allclose(sins.regular_resampled().t, regular_times)

        np.testing.assert_allclose(
            sins.regular_resampled().y,
            np.sin(regular_times),
            atol=1e-7,
            rtol=3e-6,
        )

        sins.regular_resample()

        np.testing.assert_allclose(sins.t, regular_times)
        np.testing.assert_allclose(
            sins.y, np.sin(regular_times), atol=1e-7, rtol=3e-6
        )

        sins = self.TS.copy()

        two_times = np.linspace(0, 2 * np.pi, 200)
        dt = two_times[1] - two_times[0]

        np.testing.assert_allclose(
            sins.fixed_frequency_resampled(1 / dt).t, two_times
        )
        np.testing.assert_allclose(
            sins.fixed_frequency_resampled(1 / dt).y,
            np.sin(two_times),
            atol=1e-7,
            rtol=3e-6,
        )

        np.testing.assert_allclose(
            sins.fixed_timestep_resampled(dt).t, two_times
        )
        np.testing.assert_allclose(
            sins.fixed_timestep_resampled(dt).y,
            np.sin(two_times),
            atol=1e-7,
            rtol=3e-6,
        )

        with self.assertRaises(ValueError):
            sins.fixed_timestep_resampled(50)

        with self.assertRaises(ValueError):
            sins.fixed_frequency_resampled(1e-5)

        sins2 = sins.copy()
        sins.fixed_timestep_resample(dt)
        sins2.fixed_frequency_resample(1 / dt)

        np.testing.assert_allclose(sins.t, two_times)
        np.testing.assert_allclose(
            sins.y, np.sin(two_times), atol=1e-7, rtol=3e-6
        )
        np.testing.assert_allclose(sins2.t, two_times)
        np.testing.assert_allclose(
            sins2.y, np.sin(two_times), atol=1e-7, rtol=3e-6
        )

        # Test resample with piecewise_constant

        # For this, we prepare an array with only two values, and resample
        # on four points.
        res = ts.TimeSeries([1, 2], [10, 0])
        res.resample([1, 1.1, 1.9, 2], piecewise_constant=True)
        np.testing.assert_allclose(res.y, np.array([10, 10, 0, 0]))

    def test_integrate(self):
        times_long = np.linspace(0, 2 * np.pi, 10000)
        values_long = np.sin(times_long)
        TS = ts.TimeSeries(times_long, values_long)
        TS_c = ts.TimeSeries(times_long, values_long + 1j * values_long)

        np.testing.assert_allclose(
            TS.integrated().y, 1 - np.cos(times_long), atol=1e-7
        )

        np.testing.assert_allclose(
            TS_c.integrated().y,
            1 - np.cos(times_long) + 1j * (1 - np.cos(times_long)),
            atol=1e-7,
        )

        sins = TS.copy()
        sins.integrate()

        np.testing.assert_allclose(
            sins.y, 1 - np.cos(times_long), rtol=1e-7, atol=1e-7
        )

        # Masked data
        with self.assertRaises(RuntimeError):
            self.TS_cm.integrated()

    def test_differentiate(self):
        times = np.linspace(0, 2 * np.pi, 1000)
        values = np.sin(times)

        higher_res_TS = ts.TimeSeries(times, values)
        higher_res_TS_c = ts.TimeSeries(times, values + 1j * values)

        np.testing.assert_allclose(
            higher_res_TS.differentiated().y, np.cos(times), rtol=1e-3
        )

        np.testing.assert_allclose(
            higher_res_TS.differentiated(2).y,
            -np.sin(times),
            rtol=5e-2,
            atol=5e-2,
        )

        np.testing.assert_allclose(
            higher_res_TS_c.differentiated().y,
            np.cos(times) + 1j * np.cos(times),
            rtol=1e-3,
            atol=1e-3,
        )

        sins = higher_res_TS.copy()
        sins.differentiate()

        np.testing.assert_allclose(sins.y, np.cos(times), rtol=1e-3)

        sins = higher_res_TS.copy()

        with self.assertRaises(ValueError):
            sins.spline_differentiated(8)

        # The boundaries are not accurate
        np.testing.assert_allclose(
            higher_res_TS.spline_differentiated().y,
            np.cos(times),
            rtol=1e-3,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            higher_res_TS.spline_differentiated(2).y,
            -np.sin(times),
            rtol=1e-3,
            atol=1e-3,
        )

        np.testing.assert_allclose(
            higher_res_TS_c.spline_differentiated().y,
            np.cos(times) + 1j * np.cos(times),
            rtol=1e-3,
            atol=1e-3,
        )

        sins.spline_differentiate()

        np.testing.assert_allclose(sins.y, np.cos(times), rtol=1e-3)

        with self.assertRaises(RuntimeError):
            self.TS_cm.differentiated()

    def test_remove_duplicated_iters(self):
        t = np.array([1, 2, 3, 4, 2, 3])
        y = np.array([0, 0, 0, 0, 0, 0])

        self.assertEqual(
            ts.remove_duplicated_iters(t, y),
            ts.TimeSeries([1, 2, 3], [0, 0, 0]),
        )

    def test_time_unit_change(self):
        sins = ts.TimeSeries(self.times, self.values)

        new_times = np.linspace(0, np.pi, 100)
        new_times_inverse = np.linspace(0, 6 * np.pi, 100)

        np.testing.assert_allclose(sins.time_unit_changed(2).t, new_times)

        np.testing.assert_allclose(
            sins.time_unit_changed(3, inverse=True).t, new_times_inverse
        )

        sins.time_unit_change(2)

        np.testing.assert_allclose(sins.t, new_times)

        sins.time_unit_change(6, inverse=True)

        np.testing.assert_allclose(sins.t, new_times_inverse)

        two_times = np.linspace(0, 4 * np.pi, 100)

        sins = self.TS.copy()

        sins_half_frequency = ts.TimeSeries(two_times, np.sin(0.5 * two_times))

        self.assertEqual(sins.redshifted(1), sins_half_frequency)

        sins.redshift(1)

        self.assertEqual(sins, sins_half_frequency)

    def test_combine_ts(self):
        times1 = np.linspace(0, 2 * np.pi, 100)
        sins1 = np.sin(times1)
        times2 = np.linspace(np.pi, 3 * np.pi, 100)
        coss1 = np.cos(times2)

        # A sine wave + half a cos wave
        expected_early = np.append(sins1, np.cos(times2[50:]))
        expected_late = np.append(sins1[:50], np.cos(times2))

        ts1 = ts.TimeSeries(times1, sins1)
        ts2 = ts.TimeSeries(times2, coss1)

        np.testing.assert_allclose(
            ts.combine_ts([ts1, ts2], prefer_late=False).y, expected_early
        )

        np.testing.assert_allclose(ts.combine_ts([ts1, ts2]).y, expected_late)

        # Here we test two timeseries with same tmin
        times4 = np.linspace(0, 2 * np.pi, 100)
        times5 = np.linspace(0, 3 * np.pi, 100)
        sins4 = np.sin(times4)
        coss5 = np.sin(times5)

        ts4 = ts.TimeSeries(times4, sins4)
        ts5 = ts.TimeSeries(times5, coss5)

        np.testing.assert_allclose(
            ts.combine_ts([ts1, ts2], prefer_late=False).y, expected_early
        )

        np.testing.assert_allclose(
            ts.combine_ts([ts4, ts5], prefer_late=True).y, coss5
        )

    def test_resample_common(self):
        # Test with resample=False
        ts_short1 = ts.TimeSeries([1, 2, 3, 4, 5], [11, 12, 13, 14, 15])
        ts_short2 = ts.TimeSeries([0, 2, 3, 5], [20, 22, 23, 25])
        ts_short3 = ts.TimeSeries([0, 6, 7, 8], [20, 26, 27, 28])

        new_ts_short1, new_ts_short2 = series.sample_common(
            [ts_short1, ts_short2]
        )

        np.testing.assert_allclose(new_ts_short1.t, [2, 3, 5])
        np.testing.assert_allclose(new_ts_short2.t, [2, 3, 5])
        np.testing.assert_allclose(new_ts_short1.y, [12, 13, 15])
        np.testing.assert_allclose(new_ts_short2.y, [22, 23, 25])

        # Test no common point
        with self.assertRaises(ValueError):
            series.sample_common([ts_short1, ts_short2, ts_short3])

        times1 = np.linspace(0, 2 * np.pi, 5000)
        times2 = np.linspace(np.pi, 3 * np.pi, 5000)
        times3 = np.linspace(np.pi, 2 * np.pi, 5000)
        sins1 = np.sin(times1)
        sins2 = np.sin(times2)
        sins3 = np.sin(times3)

        ts1 = ts.TimeSeries(times1, sins1)
        ts2 = ts.TimeSeries(times2, sins2)

        new_ts1, new_ts2 = series.sample_common([ts1, ts2], resample=True)

        np.testing.assert_allclose(new_ts1.y, sins3, atol=1e-14)

        # Test with piecewise_constant = True

        new_ts1_c, new_ts2_c = series.sample_common(
            [ts1, ts2], piecewise_constant=True, resample=True
        )

        # The accuracy is not as great
        np.testing.assert_allclose(new_ts1_c.y, sins3, rtol=1e-3, atol=1e-3)

        # Test a case in which there's no resampling
        newer_ts1, newer_ts2 = series.sample_common(
            [new_ts1, new_ts2], resample=True
        )
        np.testing.assert_allclose(new_ts1.y, sins3, atol=1e-14)

        # Case with different lengths
        times1_longer = np.append(-1, np.linspace(0, 2 * np.pi, 5000))
        sins1_longer = np.sin(times1_longer)
        ts1_longer = ts.TimeSeries(times1_longer, sins1_longer)

        ts1_res, ts2_res = series.sample_common(
            [ts2, ts1_longer], resample=True
        )
        np.testing.assert_allclose(ts1_res.y, sins3, atol=1e-14)

    def test_windows(self):
        ones = ts.TimeSeries(self.times, np.ones_like(self.times))
        tuk_array = signal.tukey(len(ones), 0.5)
        ham_array = signal.hamming(len(ones))
        black_array = signal.blackman(len(ones))

        np.testing.assert_allclose(ones.tukey_windowed(0.5).y, tuk_array)

        np.testing.assert_allclose(ones.hamming_windowed().y, ham_array)

        np.testing.assert_allclose(ones.blackman_windowed().y, black_array)

        new_ones = ones.copy()
        new_ones.tukey_window(0.5)
        np.testing.assert_allclose(new_ones.y, tuk_array)

        new_ones = ones.copy()
        new_ones.hamming_window()
        np.testing.assert_allclose(new_ones.y, ham_array)

        new_ones = ones.copy()
        new_ones.blackman_window()
        np.testing.assert_allclose(new_ones.y, black_array)

        # Test window directly
        new_ones = ones.copy()
        # Error for window not implemented
        with self.assertRaises(ValueError):
            new_ones.window("planck")

        # Error for window in wrong format
        with self.assertRaises(TypeError):
            new_ones.window([1, 2])

        # Window called as string
        new_ones.window("blackman")
        np.testing.assert_allclose(new_ones.y, black_array)

        # Test window on not regularly sampled series
        log_times = np.logspace(-2, -1, 10)
        new_ones_log = ts.TimeSeries(log_times, np.ones_like(log_times))
        with self.assertWarns(RuntimeWarning):
            new_ones_log.window("blackman")

        new_ones_log_resampled = ts.TimeSeries(
            log_times, np.ones_like(log_times)
        ).regular_resampled()
        new_ones_log_resampled.window("blackman")

        np.testing.assert_allclose(new_ones_log.t, new_ones_log_resampled.t)
        np.testing.assert_allclose(new_ones_log.y, new_ones_log_resampled.y)

    def test_savgol_smooth(self):
        # Here I just test that I am correctly calling the filter
        # from scipy

        expected_y = signal.savgol_filter(self.values, 11, 3)

        np.testing.assert_allclose(
            self.TS.savgol_smoothed(11, 3).y, expected_y
        )

        np.testing.assert_allclose(
            self.TS_c.savgol_smoothed(11, 3).y,
            expected_y + 1j * expected_y,
        )

        # dt = 0.063...  if tsmooth = 0.63, then window size is 11
        np.testing.assert_allclose(
            self.TS.savgol_smoothed_time(0.63, 3).y, expected_y
        )

        np.testing.assert_allclose(
            self.TS_c.savgol_smoothed_time(0.63, 3).y,
            expected_y + 1j * expected_y,
        )

        # Test non regularly sampled
        with self.assertWarns(RuntimeWarning):
            tts = self.TS.copy()
            tts.t[1] *= 1.01
            tts.savgol_smoothed_time(0.63, 3)

        sins = self.TS.copy()
        sins.savgol_smooth(11, 3)

        np.testing.assert_allclose(sins.y, expected_y)

        sins = self.TS.copy()
        sins.savgol_smooth_time(0.63, 3)

        np.testing.assert_allclose(sins.y, expected_y)

        with self.assertRaises(RuntimeError):
            self.TS_cm.savgol_smoothed(1, 3)

    def test_mask(self):
        # Clean data
        self.assertCountEqual(self.TS.mask, [False] * len(self.TS))

        # Masked data
        self.assertCountEqual(self.TS_cm.mask, self.TS_cm.y.mask)

    def test_mask_remove(self):
        # Test not masked data
        ts_nomask = self.TS.copy()
        ts_nomask.mask_remove()

        np.testing.assert_allclose(ts_nomask.y, self.values)

        t = np.linspace(-1, 1, 99)
        ts_mask = ts.TimeSeries(t, np.ma.masked_less_equal(t, 0))

        ts_mask.mask_remove()

        np.testing.assert_allclose(t[50:], ts_mask.y)

        # When we remove a mask, the series should be non masked
        self.assertFalse(ts_mask.is_masked())

    def test_mask_apply(self):
        ts_mask = ts.TimeSeries(
            self.times, np.ma.masked_less_equal(self.times, 1)
        )
        ts_nomask = ts.TimeSeries(self.times, self.times)

        ts_nomask.mask_apply(ts_mask.mask)

        self.assertEqual(ts_mask, ts_nomask)

        # Check with series already masked, by applying a second
        # mask

        ts_mask05 = ts.TimeSeries(
            self.times, np.ma.masked_less_equal(self.times, 0.5)
        )

        ts_nomask.mask_apply(ts_mask05.mask)

        self.assertEqual(ts_mask05, ts_nomask)

    def test_unfold_phase(self):
        y = np.append(
            np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2 * np.pi, 100)
        )

        yexp = np.append(
            np.linspace(0, 2 * np.pi, 100),
            np.linspace(0, 2 * np.pi, 100) + 2 * np.pi,
        )
        np.testing.assert_allclose(ts.unfold_phase(y), yexp)

        exp = ts.TimeSeries(self.times, np.exp(1j * self.times))

        np.testing.assert_allclose(exp.unfolded_phase().y, self.times)

        # test t_of_zero_phase
        # The phase at 1 is 1, so everything has to be scaled down by 1
        np.testing.assert_allclose(
            exp.unfolded_phase(t_of_zero_phase=1).y, self.times - 1
        )

        # deriv is trivial...
        deriv = np.gradient(self.times, self.times)

        np.testing.assert_allclose(
            exp.phase_angular_velocity(use_splines=False).y, deriv
        )

        np.testing.assert_allclose(exp.phase_angular_velocity().y, deriv)

        np.testing.assert_allclose(
            exp.phase_frequency().y, deriv / (2 * np.pi)
        )

        smoothed_deriv = signal.savgol_filter(deriv, 11, 3)
        # 0.63 corresponds to 11 points
        np.testing.assert_allclose(
            exp.phase_frequency(tsmooth=0.63).y,
            smoothed_deriv / (2 * np.pi),
        )

    def test_to_FrequencySeries(self):
        # Masked data
        with self.assertRaises(RuntimeError):
            self.TS_cm.to_FrequencySeries()

        # Test complex
        dt = self.times[1] - self.times[0]
        freq = np.fft.fftfreq(len(self.values), d=dt)
        freq = np.fft.fftshift(freq)
        fft = np.fft.fft(self.values + 1j * self.values)
        fft = np.fft.fftshift(fft) * dt

        fs = self.TS_c.to_FrequencySeries()

        # Test warning for non reqularly sampled
        with self.assertWarns(RuntimeWarning):
            tts = self.TS.copy()
            tts.t[1] *= 1.01
            tts.to_FrequencySeries()

        np.testing.assert_allclose(fs.f, freq)
        np.testing.assert_allclose(fs.fft, fft)

        # Test real
        rfreq = np.fft.rfftfreq(len(self.values), d=dt)
        rfft = np.fft.rfft(self.values) * dt

        rfs = self.TS.to_FrequencySeries()

        np.testing.assert_allclose(rfs.f, rfreq)
        np.testing.assert_allclose(rfs.fft, rfft)
