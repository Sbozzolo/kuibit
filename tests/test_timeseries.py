#!/usr/bin/env python3
"""Tests for postcactus.timeseries
"""

import unittest
from postcactus import timeseries as ts
import numpy as np
from scipy import signal
import os


class TestTimeseries(unittest.TestCase):

    def setUp(self):
        self.times = np.linspace(0, 2 * np.pi, 100)
        self.values = np.sin(self.times)

        self.TS = ts.TimeSeries(self.times, self.values)
        # Complex
        self.TS_c = ts.TimeSeries(self.times,
                                  self.values + 1j * self.values)

    def test_init(self):
        # Check that errors are thrown if:
        # 1. There is a mismatch between t and y
        # 2. The timeseries is empty
        # 3. The time is not monotonically increasing

        # 1
        times = np.linspace(0, 2 * np.pi, 100)
        values = np.array([1, 2, 3])

        with self.assertRaises(ValueError):
            t = ts.TimeSeries(times, values)

        # 2
        times = np.array([])
        values = np.array([])

        with self.assertRaises(ValueError):
            t = ts.TimeSeries(times, values)

        # 3
        times = np.linspace(2 * np.pi, 0, 100)
        values = np.sin(times)

        with self.assertRaises(ValueError):
            t = ts.TimeSeries(times, values)

        # Let's check that we can instantiate TimeSeries with 1 element
        # It shouls throw a warning because it cannot compute the spline
        with self.assertWarns(Warning):
            scalar = ts.TimeSeries(0, 0)

        self.assertEqual(scalar.y, 0)

    def test_len(self):
        self.assertEqual(len(self.TS), 100)

    def test_tmin_tmax_length_dt(self):
        self.assertAlmostEqual(self.TS.tmin, 0)
        self.assertAlmostEqual(self.TS.tmax, 2 * np.pi)

        times = np.linspace(1, 2, 100)

        sins = ts.TimeSeries(times, self.values)

        self.assertAlmostEqual(sins.time_length, 1)

        self.assertAlmostEqual(self.TS.dt,
                               self.TS.tmax / (len(self.TS) - 1))

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
            out = self.TS._apply_binary(cos, np.add)

        # 1b
        times = np.linspace(2, 3 * np.pi, 100)
        values = np.cos(times)
        cos = ts.TimeSeries(times, values)

        with self.assertRaises(ValueError):
            out = self.TS._apply_binary(cos, np.add)

        # 2
        with self.assertRaises(TypeError):
            out = self.TS._apply_binary("str", np.add)

    def test_add(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        out = self.TS + ts.TimeSeries(times, values)
        self.assertTrue(np.allclose(out.y, 2 * np.sin(times)))

        # Scalar
        out = self.TS + 1
        self.assertTrue(np.allclose(out.y, 1 + np.sin(times)))

        out = 1 + self.TS
        self.assertTrue(np.allclose(out.y, 1 + np.sin(times)))

        # Test iadd
        out += 1
        self.assertTrue(np.allclose(out.y, 2 + np.sin(times)))

        out += self.TS
        self.assertTrue(np.allclose(out.y, 2 + 2*np.sin(times)))

    def test_sub(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        out = self.TS - ts.TimeSeries(times, values)
        self.assertTrue(np.allclose(out.y, 0))

        # Scalar
        out = self.TS - 1
        self.assertTrue(np.allclose(out.y, np.sin(times) - 1))

        out = 1 - self.TS
        self.assertTrue(np.allclose(out.y, 1 - np.sin(times)))

        # Test isub
        out -= 1
        self.assertTrue(np.allclose(out.y, -np.sin(times)))

        out -= self.TS
        self.assertTrue(np.allclose(out.y, -2*np.sin(times)))

    def test_neg(self):
        self.assertTrue(np.allclose((-self.TS).y, -np.sin(self.times)))

    def test_abs(self):
        self.assertTrue(np.allclose(abs(self.TS).y,
                                    np.abs(np.sin(self.times))))
        self.assertTrue(np.allclose(np.abs(self.TS).y,
                                    np.abs(np.sin(self.times))))
        self.assertTrue(np.allclose(self.TS.abs().y,
                                    np.abs(np.sin(self.times))))

    def test_mul(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        out = self.TS * ts.TimeSeries(times, values)
        self.assertTrue(np.allclose(out.y, np.sin(times)**2))

        # Scalar
        out = self.TS * 3
        self.assertTrue(np.allclose(out.y, np.sin(times) * 3))

        out = 3 * self.TS
        self.assertTrue(np.allclose(out.y, np.sin(times) * 3))

        # Test imul
        out *= 3
        self.assertTrue(np.allclose(out.y, 9 * np.sin(times)))

        out *= self.TS
        self.assertTrue(np.allclose(out.y, 9*np.sin(times)**2))

    def test_div(self):
        # Errors are tested by test_apply_binary

        with self.assertRaises(ValueError):
            out = self.TS / 0

        # Let's avoid division by 0

        times = np.linspace(np.pi / 3, np.pi / 2, 100)
        values = np.sin(times)

        sins = ts.TimeSeries(times, values)

        out = sins / sins
        self.assertTrue(np.allclose(out.y, 1))

        # Scalar
        out = sins / 3
        self.assertTrue(np.allclose(out.y, np.sin(times) / 3))

        # Test itruediv
        out /= 2
        self.assertTrue(np.allclose(out.y, np.sin(times)/6))

        out /= sins
        self.assertTrue(np.allclose(out.y, 1/6))

    def test_power(self):
        # Errors are tested by test_apply_binary

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.array([2]*len(times))

        out = self.TS ** ts.TimeSeries(times, values)
        self.assertTrue(np.allclose(out.y, np.sin(times)**2))

        # Scalar
        out = self.TS ** 2
        self.assertTrue(np.allclose(out.y, np.sin(times) ** 2))

        # Test ipow
        out **= 2
        self.assertTrue(np.allclose(out.y, np.sin(times)**4))

        out **= ts.TimeSeries(times, values)
        self.assertTrue(np.allclose(out.y, np.sin(times)**8))

    def test_eq(self):

        times = np.linspace(0, 2 * np.pi, 100)
        values = np.sin(times)

        sins = ts.TimeSeries(times, values)

        self.assertFalse(sins == 1)
        self.assertTrue(self.TS == sins)

    def test_is_complex(self):
        self.assertFalse(self.TS.is_complex())
        self.assertTrue(self.TS_c.is_complex())

    def test_unary_functions(self):

        def test_f(f):
            times = np.linspace(np.pi / 3, np.pi / 2, 100)
            values = np.array([0.5]*len(times))

            out = f(ts.TimeSeries(times, values))
            self.assertTrue(np.allclose(out.y, f(values)))

        def test_f2(f):
            times = np.linspace(np.pi / 3, np.pi / 2, 100)
            values = np.array([1.5]*len(times))

            out = f(ts.TimeSeries(times, values))
            self.assertTrue(np.allclose(out.y, f(values)))

        for f in [np.abs,
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
                  np.log10,
                  np.conj,
                  np.conjugate]:
            with self.subTest(f=f):
                test_f(f)

        # Different domain
        for f in [np.arccosh]:
            with self.subTest(f=f):
                test_f2(f)

        # Different call method
        out = (self.TS).real()
        self.assertTrue(np.allclose(out.y, np.sin(self.times)))
        out = (self.TS).imag()
        self.assertTrue(np.allclose(out.y, 0))

    def test_zero_pad(self):

        # Check invalid number of points

        with self.assertRaises(ValueError):
            out = self.TS.zero_pad(50)

        # Check unevenly-space time
        times = np.logspace(0, 1, 100)
        sins = ts.TimeSeries(times, self.values)

        with self.assertRaises(ValueError):
            out = sins.zero_pad(200)

        sins2 = ts.TimeSeries(self.times, self.values)

        sins2.zero_pad(200)

        self.assertTrue(np.allclose(sins2.y[100:], np.zeros(100)), 1)

        # Check if the sequence is still equispaced in time
        dts = np.diff(sins2.t)
        dt0 = dts[0]
        self.assertTrue(np.allclose(dts, dt0))

    def test_mean_remove(self):

        # Check unevenly-space time
        times = np.logspace(0, 1, 100)
        sins = ts.TimeSeries(times, self.values + 1)

        self.assertTrue(np.allclose(sins.mean_removed().y, self.values))

        sins.mean_remove()

        self.assertTrue(np.allclose(sins.y, self.values))

    def test_copy(self):
        tscopy = self.TS.copy()
        tscopyc = self.TS_c.copy()
        self.assertEqual(self.TS, tscopy)
        self.assertEqual(self.TS_c, tscopyc)

    def test_time_shift(self):

        times = np.logspace(0, 1, 100)
        sins = ts.TimeSeries(times, self.values)

        self.assertTrue(np.allclose(sins.time_shifted(1).t,
                                    times + 1))

        sins.time_shift(1)

        self.assertTrue(np.allclose(sins.t, times + 1))

    def test_phase_shift(self):

        sins = ts.TimeSeries(self.times, self.values)

        self.assertTrue(np.allclose(sins.phase_shifted(np.pi/2).y,
                                    1j * self.values))
        sins.phase_shift(np.pi/2)

        self.assertTrue(np.allclose(sins.y, 1j * self.values))

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

        self.assertTrue(np.allclose(loaded_sins[0], times))
        self.assertTrue(np.allclose(loaded_sins[1], np.sin(times)))

        self.assertTrue(np.allclose(loaded_compl[0], times))
        self.assertTrue(np.allclose(loaded_compl[1], np.sin(times)))
        self.assertTrue(np.allclose(loaded_compl[2], np.sin(times)))

    def test_nans_remove(self):

        values = self.values[:]
        values[0] = np.inf
        values[-1] = np.nan
        sins = ts.TimeSeries(self.times, values)

        self.assertTrue(np.allclose(sins.nans_removed().y,
                                    values[1:-1]))
        self.assertTrue(np.allclose(sins.nans_removed().t,
                                    self.times[1:-1]))

        sins.nans_remove()

        self.assertTrue(np.allclose(sins.y, values[1:-1]))
        self.assertTrue(np.allclose(sins.t, self.times[1:-1]))

    def test_make_spline_call(self):

        # Cannot make a spline with 1 point
        with self.assertRaises(ValueError):
            sins = self.TS.copy()
            # We are checking if the number of points change
            # after initialization because we've already checked
            # at initialization
            sins.t = np.array([0])
            sins.y = np.array([0])
            sins._make_spline()

        # Check that spline reproduce data
        self.assertTrue(np.allclose(self.TS(self.times),
                                    self.values))

        self.assertTrue(np.allclose(self.TS(np.pi/2), 1))

        self.assertTrue(np.allclose(self.TS_c(self.times),
                                    self.values + 1j * self.values))
        self.assertTrue(np.allclose(self.TS(self.times),
                                    self.values))

        # Does the spline update?
        # Let's test with a method that changes the timeseries
        sins = ts.TimeSeries(self.times, self.values + 1)
        self.assertTrue(np.allclose(sins(self.times),
                                    self.values + 1))

        sins.mean_remove()

        self.assertTrue(np.allclose(sins(self.times),
                                    self.values))

        with self.assertRaises(ValueError):
            sins(-1)

    def test_resample(self):

        new_times = np.array([float(1e-5 * i**2) for i in range(0, 100)])
        sins = self.TS.copy()

        self.assertTrue(np.allclose(sins.resampled(new_times).y,
                                    np.sin(new_times)))
        self.assertTrue(np.allclose(sins.resampled(new_times).t,
                                    new_times))

        sins.resample(new_times)

        self.assertTrue(np.allclose(sins.y, np.sin(new_times)))
        self.assertTrue(np.allclose(sins.t, new_times))

        # Test regular_sample using sins, that now is unevenly
        # sampled
        regular_times = np.linspace(0, new_times[-1], 100)

        self.assertTrue(np.allclose(sins.regular_resampled().t,
                                    regular_times))

        self.assertTrue(np.allclose(sins.regular_resampled().y,
                                    np.sin(regular_times)))

        sins.regular_resample()

        self.assertTrue(np.allclose(sins.t, regular_times))
        self.assertTrue(np.allclose(sins.y, np.sin(regular_times)))

        sins = self.TS.copy()

        two_times = np.linspace(0, 2 * np.pi, 200)
        dt = two_times[1] - two_times[0]

        self.assertTrue(
            np.allclose(sins.fixed_frequency_resampled(1 / dt).t, two_times))
        self.assertTrue(
            np.allclose(
                sins.fixed_frequency_resampled(1 / dt).y, np.sin(two_times)))

        self.assertTrue(
            np.allclose(sins.fixed_timestep_resampled(dt).t, two_times))
        self.assertTrue(
            np.allclose(
                sins.fixed_timestep_resampled(dt).y, np.sin(two_times)))

        with self.assertRaises(ValueError):
            sins.fixed_timestep_resampled(50)

        with self.assertRaises(ValueError):
            sins.fixed_frequency_resampled(1e-5)

        sins2 = sins.copy()
        sins.fixed_timestep_resample(dt)
        sins2.fixed_frequency_resample(1/dt)

        self.assertTrue(np.allclose(sins.t, two_times))
        self.assertTrue(np.allclose(sins.y, np.sin(two_times)))
        self.assertTrue(np.allclose(sins2.t, two_times))
        self.assertTrue(np.allclose(sins2.y, np.sin(two_times)))

    def test_integrate(self):

        self.assertTrue(np.allclose(self.TS.integrated().y,
                                    1 - np.cos(self.times),
                                    atol=1e-3))

        self.assertTrue(np.allclose(self.TS_c.integrated().y,
                                    1 - np.cos(self.times)
                                    + 1j * (1 - np.cos(self.times)),
                                    atol=1e-3))

        sins = self.TS.copy()
        sins.integrate()

        self.assertTrue(np.allclose(sins.y,
                                    1 - np.cos(self.times),
                                    atol=1e-3))

    def test_derive(self):

        self.assertTrue(np.allclose(self.TS.derived().y,
                                    np.cos(self.times),
                                    atol=1e-3))

        self.assertTrue(np.allclose(self.TS.derived(2).y,
                                    -np.sin(self.times),
                                    atol=5e-2))

        self.assertTrue(np.allclose(self.TS_c.derived().y,
                                    np.cos(self.times)
                                    + 1j * np.cos(self.times),
                                    atol=1e-3))

        sins = self.TS.copy()
        sins.derive()

        self.assertTrue(np.allclose(sins.y,
                                    np.cos(self.times),
                                    atol=1e-3))

        sins = self.TS.copy()

        with self.assertRaises(ValueError):
            sins.spline_derived(8)

        # The boundaries are not accurate
        self.assertTrue(np.allclose(self.TS.spline_derived().y,
                                    np.cos(self.times),
                                    atol=1e-3))

        self.assertTrue(np.allclose(self.TS.spline_derived(2).y,
                                    -np.sin(self.times),
                                    atol=5e-2))

        self.assertTrue(np.allclose(self.TS_c.spline_derived().y,
                                    np.cos(self.times)
                                    + 1j * np.cos(self.times),
                                    atol=1e-3))

        sins.spline_derive()

        self.assertTrue(np.allclose(sins.y,
                                    np.cos(self.times),
                                    atol=1e-3))

    def test_remove_duplicate_iters(self):

        t = np.array([1, 2, 3, 4, 2, 3])
        y = np.array([0, 0, 0, 0, 0, 0])

        # There is warning that is thrown (unrelated to the test)
        with self.assertWarns(Warning):
            self.assertEqual(ts.remove_duplicate_iters(t, y),
                             ts.TimeSeries([1, 2, 3],
                                           [0, 0, 0]))

    def test_time_unit_change(self):

        sins = ts.TimeSeries(self.times, self.values)

        new_times = np.linspace(0, np.pi, 100)
        new_times_inverse = np.linspace(0, 6 * np.pi, 100)

        self.assertTrue(np.allclose(sins.time_unit_changed(2).t,
                                    new_times))

        self.assertTrue(np.allclose(
            sins.time_unit_changed(3, inverse=True).t,
            new_times_inverse))

        sins.time_unit_change(2)

        self.assertTrue(np.allclose(sins.t, new_times))

        sins.time_unit_change(6, inverse=True)

        self.assertTrue(np.allclose(sins.t, new_times_inverse))

        two_times = np.linspace(0, 4 * np.pi, 100)

        sins = self.TS.copy()

        sins_half_frequency = ts.TimeSeries(two_times,
                                            np.sin(0.5*two_times))

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

        self.assertTrue(np.allclose(ts.combine_ts([ts1, ts2],
                                                  prefer_late=False).y,
                                    expected_early))

        self.assertTrue(np.allclose(ts.combine_ts([ts1, ts2]).y,
                                    expected_late))

        # Here we test two timeseries with same tmin
        times4 = np.linspace(0, 2 * np.pi, 100)
        times5 = np.linspace(0, 3 * np.pi, 100)
        sins4 = np.sin(times4)
        coss5 = np.sin(times5)

        ts4 = ts.TimeSeries(times4, sins4)
        ts5 = ts.TimeSeries(times5, coss5)

        self.assertTrue(np.allclose(ts.combine_ts([ts1, ts2],
                                                  prefer_late=False).y,
                                    expected_early))

        self.assertTrue(np.allclose(ts.combine_ts([ts4, ts5],
                                                  prefer_late=True).y,
                                    coss5))

    def test_resample_common(self):

        times1 = np.linspace(0, 2 * np.pi, 100)
        times2 = np.linspace(np.pi, 3 * np.pi, 100)
        times3 = np.linspace(np.pi, 2 * np.pi, 100)
        sins1 = np.sin(times1)
        sins2 = np.sin(times2)
        sins3 = np.sin(times3)

        ts1 = ts.TimeSeries(times1, sins1)
        ts2 = ts.TimeSeries(times2, sins2)

        new_ts1, new_ts2 = ts.sample_common([ts1, ts2])

        self.assertTrue(np.allclose(new_ts1.y, sins3))

    def test_windows(self):

        ones = ts.TimeSeries(self.times, np.ones_like(self.times))
        tuk_array = signal.tukey(len(ones), 0.5)
        ham_array = signal.hamming(len(ones))
        black_array = signal.blackman(len(ones))

        self.assertTrue(np.allclose(ones.tukey_windowed(0.5).y,
                                    tuk_array))

        self.assertTrue(np.allclose(ones.hamming_windowed().y,
                                    ham_array))

        self.assertTrue(np.allclose(ones.blackman_windowed().y,
                                    black_array))

        new_ones = ones.copy()
        new_ones.tukey_window(0.5)
        self.assertTrue(np.allclose(new_ones.y, tuk_array))

        new_ones = ones.copy()
        new_ones.hamming_window()
        self.assertTrue(np.allclose(new_ones.y, ham_array))

        new_ones = ones.copy()
        new_ones.blackman_window()
        self.assertTrue(np.allclose(new_ones.y, black_array))

    def test_savgol_smooth(self):

        # Here I just test that I am correctly calling the filter
        # from scipy

        expected_y = signal.savgol_filter(self.values, 11, 3)

        self.assertTrue(np.allclose(self.TS.savgol_smoothed(11, 3).y,
                                    expected_y))

        self.assertTrue(np.allclose(self.TS_c.savgol_smoothed(11, 3).y,
                                    expected_y + 1j * expected_y))

        # dt = 0.063...  if tsmooth = 0.63, then window size is 11
        self.assertTrue(np.allclose(self.TS.savgol_smoothed_time(0.63, 3).y,
                                    expected_y))

        self.assertTrue(np.allclose(self.TS_c.savgol_smoothed_time(0.63, 3).y,
                                    expected_y + 1j * expected_y))

        sins = self.TS.copy()
        sins.savgol_smooth(11, 3)

        self.assertTrue(np.allclose(sins.y, expected_y))

        sins = self.TS.copy()
        sins.savgol_smooth_time(0.63, 3)

        self.assertTrue(np.allclose(sins.y, expected_y))

    def test_unfold_phase(self):

        y = np.append(np.linspace(0, 2 * np.pi, 100),
                      np.linspace(0, 2 * np.pi, 100))

        yexp = np.append(np.linspace(0, 2 * np.pi, 100),
                         np.linspace(0, 2 * np.pi, 100) + 2 * np.pi)
        self.assertTrue(np.allclose(ts.unfold_phase(y),
                                    yexp))

        exp = ts.TimeSeries(self.times, np.exp(1j * self.times))

        self.assertTrue(np.allclose(exp.unfolded_phase().y,
                                    self.times))

        # deriv is trivial...
        deriv = np.gradient(self.times, self.times)

        self.assertTrue(np.allclose(
            exp.phase_angular_velocity(use_splines=False).y,
            deriv))

        self.assertTrue(np.allclose(exp.phase_angular_velocity().y,
                                    deriv))

        self.assertTrue(np.allclose(exp.phase_frequency().y,
                                    deriv / (2 * np.pi)))

        smoothed_deriv = signal.savgol_filter(deriv, 11, 3)
        self.assertTrue(np.allclose(exp.phase_frequency(tsmooth=0.63).y,
                                    smoothed_deriv / (2 * np.pi)))

    def test_to_FrequencySeries(self):

        dt = self.times[1] - self.times[0]
        freq = np.fft.fftfreq(len(self.values), d=dt)
        freq = np.fft.fftshift(freq)
        fft = np.fft.fft(self.values)
        fft = np.fft.fftshift(fft)

        fs = self.TS.to_FrequencySeries()

        self.assertTrue(np.allclose(fs.f, freq))
        self.assertTrue(np.allclose(fs.fft, fft))
