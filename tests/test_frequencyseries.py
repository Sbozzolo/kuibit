#!/usr/bin/env python3
"""Tests for postcactus.frequencyseries
"""

import unittest
from postcactus import frequencyseries as fs
import numpy as np


class TestFrequencySeries(unittest.TestCase):

    def setUp(self):
        self.x = np.linspace(0, 2 * np.pi, 100)
        self.y = np.sin(self.x)
        self.dx = self.x[1] - self.x[0]

        self.f = np.fft.fftfreq(len(self.x), d=self.dx)
        self.f = np.fft.fftshift(self.f)
        self.fft = np.fft.fft(self.y)
        self.fft = np.fft.fftshift(self.fft)

        self.FS = fs.FrequencySeries(self.f, self.fft)

    def test_fmin_fmax_frange(self):

        self.assertAlmostEqual(self.FS.fmin, -7.87816968)
        self.assertAlmostEqual(self.FS.f[0], -7.87816968)
        self.assertAlmostEqual(self.FS.fmax, 7.72060629)
        self.assertAlmostEqual(self.FS.frange, 15.59877597)
        self.assertAlmostEqual(np.amax(self.FS.amp), 49.74022843)

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

        self.assertAlmostEqual(np.amax(self.FS.normalized().amp),
                               1)

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
        self.assertGreaterEqual(np.amin(np.abs(fs_copy.band_passed(0, 1).f)), 0)

        fs_copy.low_pass(2)
        self.assertLessEqual(fs_copy.fmax, 2)

        fs_copy.high_pass(0)
        self.assertGreaterEqual(np.amin(np.abs(fs_copy.f)), 0)

        fs_copy.band_pass(0.5, 1.5)

        self.assertLessEqual(fs_copy.fmax, 1.5)
        self.assertGreaterEqual(np.amin(np.abs(fs_copy.f)), 0.5)

    def test_peaks(self):

        # From a sin wave we are expecting two peaks
        [p1, p2] = self.FS.peaks()

        self.assertAlmostEqual(p1[0], -0.15756339)
        self.assertAlmostEqual(p1[1], -0.15810417)
        self.assertAlmostEqual(p1[2], 49.74022843)

        self.assertAlmostEqual(self.FS.peaks_frequencies()[0],
                               -0.15810417)

    def test_to_TimeSeries(self):

        ts = self.FS.to_TimeSeries()

        self.assertTrue(np.allclose(ts.y, self.y))
