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

"""Tests for postcactus.frequencyseries
"""

import unittest
import warnings

import numpy as np

from postcactus import frequencyseries as fs
from postcactus import timeseries as ts


class TestFrequencySeries(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 2 * np.pi, 100)
        self.y = np.sin(self.x)
        self.y_c = np.sin(self.x) + 1j * np.sin(self.x)
        self.dx = self.x[1] - self.x[0]

        self.f_c = np.fft.fftfreq(len(self.x), d=self.dx)
        self.f_c = np.fft.fftshift(self.f_c)
        self.fft_c = np.fft.fft(self.y_c) * self.dx
        self.fft_c = np.fft.fftshift(self.fft_c)

        self.f = np.fft.rfftfreq(len(self.x), d=self.dx)
        self.fft_r = np.fft.rfft(self.y) * self.dx

        self.FS = fs.FrequencySeries(self.f, self.fft_r)

        self.FS_c = fs.FrequencySeries(self.f_c, self.fft_c)

    def test_fmin_fmax_frange(self):

        self.assertAlmostEqual(self.FS.fmin, 0)
        self.assertAlmostEqual(self.FS.f[0], 0)
        self.assertAlmostEqual(self.FS.fmax, 7.87816968)
        self.assertAlmostEqual(self.FS.frange, 7.87816968)
        self.assertAlmostEqual(np.amax(self.FS.amp), 3.15683911)

    def test_setter_f(self):

        fs_copy = self.FS.copy()
        f2 = self.f * 2
        fs_copy.f = f2

        self.assertTrue(np.allclose(fs_copy.f, f2))

    def test_df(self):

        self.assertAlmostEqual(self.FS.df, self.f[1] - self.f[0])
        fs_copy = self.FS.copy()
        fs_copy.f[1] = 10

        # Unevelnly spaced
        with self.assertRaises(ValueError):
            fs_copy.df

    def test_normalize(self):

        self.assertAlmostEqual(np.amax(self.FS.normalized().amp), 1)

        fs_copy = self.FS.copy()
        fs_copy.normalize()

        self.assertAlmostEqual(np.amax(fs_copy.amp), 1)

        with self.assertRaises(ValueError):
            fs_copy.fft = np.zeros_like(fs_copy.fft)
            fs_copy.normalize()

    def test_band_pass(self):

        fs_copy = self.FS.copy()

        self.assertLessEqual(fs_copy.low_passed(1).fmax, 1)
        # abs so that we can forget about the negative frequencies
        # Otherwise we have -7.2... as fmin
        self.assertGreaterEqual(np.amin(np.abs(fs_copy.high_passed(1).f)), 1)

        self.assertLessEqual(fs_copy.band_passed(0, 1).fmax, 1)
        self.assertGreaterEqual(
            np.amin(np.abs(fs_copy.band_passed(0, 1).f)), 0
        )

        fs_copy.low_pass(2)
        self.assertLessEqual(fs_copy.fmax, 2)

        fs_copy.high_pass(0)
        self.assertGreaterEqual(np.amin(np.abs(fs_copy.f)), 0)

        fs_copy.band_pass(0.5, 1.5)

        self.assertLessEqual(fs_copy.fmax, 1.5)
        self.assertGreaterEqual(np.amin(np.abs(fs_copy.f)), 0.5)

    def test_negative_frequencies_remove(self):

        fs_copy = self.FS.copy()

        self.assertGreaterEqual(fs_copy.negative_frequencies_removed().fmin, 0)
        fs_copy.negative_frequencies_remove()
        self.assertGreaterEqual(fs_copy.fmin, 0)

    def test_peaks(self):

        # From a sin wave we are expecting two peaks
        [p1, p2] = self.FS_c.peaks()

        self.assertAlmostEqual(p1[0], -0.15756339)
        self.assertAlmostEqual(p1[1], -0.15810417)
        self.assertAlmostEqual(p1[2], 4.46444469)

        self.assertAlmostEqual(self.FS_c.peaks_frequencies()[0], -0.15810417)

    def test_to_TimeSeries(self):

        # Complex
        ts = self.FS_c.to_TimeSeries()

        self.assertTrue(np.allclose(ts.y, self.y_c))

        # real
        ts_r = self.FS.to_TimeSeries()
        self.assertTrue(np.allclose(ts_r.y, self.y))

    def test_inner_product(self):

        with self.assertRaises(TypeError):
            self.FS.inner_product(1)

        with self.assertRaises(TypeError):
            self.FS.inner_product(self.FS, noises=1)

        # fmin > fmax
        with self.assertRaises(ValueError):
            self.FS.inner_product(self.FS, fmin=1, fmax=0)

        # fmin < 0
        with self.assertRaises(ValueError):
            self.FS.inner_product(self.FS, fmin=-1)

        # To test the inner product we construct two simple linear frequency
        # series y(f) = f + 2j * f and y2(f) = 3j * f
        #
        # The inner product is y * y2^* = (6 f**2 - 3j f**2)
        # Integrated this is 2 (fmax**3 - fmin**3) - j (fmax**3 - fmin**3)
        # Taken 4 Real is 8 (fmax**3 - fmin**3)
        # First, we test with no noise

        # Small interval and many points, so that the analtical and numerical
        # predictions agree
        f = np.linspace(1, 1.2, 25000)
        fft1 = f + 2j * f
        fft2 = 3j * f

        fs1 = fs.FrequencySeries(f, fft1)
        fs2 = fs.FrequencySeries(f, fft2)

        # Test with no fmin or fmax. This means fmin = 0, fmax = 10.
        # The result should be -2 * 10**3 + j 10**3

        self.assertAlmostEqual(
            fs1.inner_product(fs2), 8 * (1.2 ** 3 - 1), places=3
        )

        # Now restrict to (fmin = 1.1, fmax = 1.15)
        self.assertAlmostEqual(
            fs1.inner_product(fs2, fmin=1.1, fmax=1.15),
            8 * (1.15 ** 3 - 1.1 ** 3),
            places=3,
        )

        # Now add a noise of f**2
        # The inner product is y * y2^* / noise= (6 - 3j)
        # Integrated it is (6 - 3j) * (fmax - fmin)
        noise = fs.FrequencySeries(f, f ** 2)

        self.assertAlmostEqual(
            fs1.inner_product(fs2, noises=noise), 4 * 6 * (1.2 - 1), places=3
        )

        # Test same_domain
        self.assertAlmostEqual(
            fs1.inner_product(fs2, noises=noise, same_domain=True),
            4 * 6 * (1.2 - 1),
            places=3,
        )

        # Test multiple noises
        # Test with twice the same noise. The output should be doubled.
        twice_noise = [noise, noise]
        self.assertAlmostEqual(
            fs1.inner_product(fs2, noises=twice_noise),
            2 * 4 * 6 * (1.2 - 1),
            places=3,
        )

    def test_overlap(self):

        # Overlap with self should be one
        self.assertAlmostEqual(self.FS.overlap(self.FS, fmin=0.5), 1)

        # Overlap with -self should be -one
        self.assertAlmostEqual(self.FS.overlap(-self.FS, fmin=0.5), -1)

        # Test with PyCBC
        num_times = 2000
        times = np.linspace(0, 20 * 2 * np.pi, num_times)
        values1 = np.sin(40 * times)
        values2 = np.sin(60 * times)

        f1_series = ts.TimeSeries(times, values1).to_FrequencySeries()
        f2_series = ts.TimeSeries(times, values2).to_FrequencySeries()

        try:

            # Test with PyCBC
            fmin = 5
            fmax = 15

            # PyCBC raises some benign warnings. We ignore them.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                from pycbc.types import timeseries as pycbcts
                from pycbc.types import frequencyseries as pycbcfs
                from pycbc.filter import overlap

            ts1_pcbc = pycbcts.TimeSeries(values1, delta_t=times[1] - times[0])
            ts2_pcbc = pycbcts.TimeSeries(values2, delta_t=times[1] - times[0])

            ov = overlap(
                ts1_pcbc,
                ts2_pcbc,
                psd=None,
                low_frequency_cutoff=fmin,
                high_frequency_cutoff=fmax,
            )

            self.assertAlmostEqual(
                f1_series.overlap(
                    f2_series, fmin=fmin, fmax=fmax, noises=None
                ),
                ov,
                places=4,
            )

            # Test with non trivial noise
            # PyCBC requires the noise to be defined on the same frequencies as the
            # data
            df_noise = ts1_pcbc.to_frequencyseries().delta_f
            f_noise = np.array(
                [i * df_noise for i in range(num_times // 2 + 1)]
            )

            # Funky looking noise
            psd_noise = np.abs(np.sin(50 * f_noise) + 0.1)
            noise_pycbc = pycbcfs.FrequencySeries(psd_noise, delta_f=df_noise)
            noise2 = fs.FrequencySeries(f_noise, psd_noise)

            ov_noise = overlap(
                ts1_pcbc,
                ts2_pcbc,
                psd=noise_pycbc,
                low_frequency_cutoff=fmin,
                high_frequency_cutoff=fmax,
            )

            self.assertAlmostEqual(
                f1_series.overlap(
                    f2_series, fmin=fmin, fmax=fmax, noises=noise2
                ),
                ov_noise,
                places=5,
            )
        except ImportError:  # pragma: no cover
            pass

    def test_load_FrequencySeries(self):

        path = "tests/tov/output-0000/static_tov/mp_Phi2_l2_m-1_r110.69.asc"
        f, fft_real, fft_imag = np.loadtxt(path).T

        ffs = fs.FrequencySeries(f, fft_real + 1j * fft_imag)
        self.assertEqual(
            ffs, fs.load_FrequencySeries(path, complex_on_two_columns=True)
        )

        path_ligo = "tests/tov/ligo_sens.dat"
        f, fft = np.loadtxt(path_ligo).T
        ffs_ligo = fs.FrequencySeries(f, fft)
        self.assertEqual(ffs_ligo, fs.load_noise_curve(path_ligo))
