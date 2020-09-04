#!/usr/bin/env python3

import unittest

import numpy as np
from scipy import signal

from postcactus import cactus_waves as cw
from postcactus import simdir as sd
from postcactus import timeseries as ts


class TestCactusWaves(unittest.TestCase):

    def test_WavesOneDet(self):

        t1 = np.linspace(0, np.pi, 100)
        t2 = np.linspace(2 * np.pi, 3 * np.pi, 100)
        y1 = np.sin(t1)
        y2 = np.sin(t2)
        ts1 = ts.TimeSeries(t1, y1)
        ts2 = ts.TimeSeries(t2, y2)
        dist1 = 100

        data1 = [(2, 2, ts1), (2, 2, ts2)]
        data2 = [(1, 1, ts1), (1, 0, ts2),
                 (1, -1, ts2)]

        gw = cw.GravitationalWavesOneDet(dist1, data1)
        em = cw.ElectromagneticWavesOneDet(dist1, data2)

        self.assertEqual(gw.l_min, 2)
        self.assertEqual(em.l_min, 1)

    def test__fixed_frequency_integrated(self):

        # First, we test the FF integration with a function (sin(x))
        # where the threshold frequency is smaller than the real frequency
        # (the period is extremely long).
        # This is a standard integration

        # Several points is better
        # However, to really reach agreement with the analytical result
        # we would need millions of points
        t = np.linspace(0, 2 * np.pi, 19000)
        y = np.sin(t)
        tts = ts.TimeSeries(t, y)
        # Dummy object (FFI is a staticmethod)
        gwdum = cw.GravitationalWavesOneDet(0, [(2, 2, tts)])

        # # The integral of sin should be -cos
        # # The period of sin(x) is 2 pi, so we pick pcut = 1e10
        # integral = gwdum._fixed_frequency_integrated(tts, 1e10)

        # self.assertTrue(np.allclose(integral.t, t))
        # self.assertTrue(np.allclose(integral.y, -np.cos(t),
        #                             atol=5e-4))

        # # The second integral should be sin(x)
        # integral2 = gwdum._fixed_frequency_integrated(tts, 1e10,
        #                                               order=2)

        # self.assertTrue(np.allclose(integral2.y, -np.sin(t),
        #                             atol=5e-4))

        # Now, let's see the opposite case in which the frequency is lower than
        # any frequencies. The output should be the same timeseries we started
        # with, rescaled by 1/(i omega_threshold). Now the factor of 1/i
        # corresponds to a rotation of pi/2, so we expect the output to be the
        # cosine. pcut = 1e-4 -> omega_threshold = 2 pi / pcut = 2 pi * 1e4
        # Hence, the timeseries is divided by 1e-4
        integral3 = gwdum._fixed_frequency_integrated(tts, 1e-4)
        self.assertTrue(np.allclose(integral3.y * 2 * np.pi * 1e4, -np.cos(t),
                                    atol=1e-3))

        # Check warning for irregularly spaced
        with self.assertWarns(RuntimeWarning):
            tts.t[1] *= 1.01
            integral4 = gwdum._fixed_frequency_integrated(tts, 1e-4)

    def test_get_strain_lm(self):

        sim = sd.SimDir("tests/tov")
        gwdir = cw.GravitationalWavesDir(sim)
        psi4 = gwdir[110.69]

        with self.assertRaises(ValueError):
            psi4.get_strain_lm(3, 3, 1)

        # Large pcut
        with self.assertRaises(ValueError):
            psi4.get_strain_lm(2, 2, 1000)

        # Test window function errors
        with self.assertRaises(ValueError):
            psi4.get_strain_lm(2, 2, 0.1, window_function=1)
        # Not implemented
        with self.assertRaises(ValueError):
            psi4.get_strain_lm(2, 2, 0.1, window_function='bubu')

        # We do not need to test the FFI, hopefully that is already tested

        # Test window = set verything to 0
        self.assertTrue(np.allclose(
            psi4.get_strain_lm(2, 2, 0.1, window_function=lambda x: 0).y,
                        0))

        psi4lm = (psi4[(2, 2)])

        # Test when window is a function
        ham_array = signal.hamming(len(psi4lm))
        ham_psi4lm = psi4lm.copy()
        ham_psi4lm.y *= ham_array
        ffi_ham = psi4._fixed_frequency_integrated(ham_psi4lm, 0.1).y,

        self.assertTrue(np.allclose(
            psi4.get_strain_lm(2, 2, 0.1, window_function=signal.hamming,
                               trim_ends=False).y, ffi_ham, atol=1e-7))

        # Test window is a string, like hamming
        self.assertTrue(np.allclose(
            psi4.get_strain_lm(2, 2, 0.1, window_function='hamming',
                               trim_ends=False).y, ffi_ham, atol=1e-7))

        # Test no window
        self.assertTrue(np.allclose(
            psi4.get_strain_lm(2, 2, 0.1, trim_ends=False).y,
            psi4._fixed_frequency_integrated(psi4lm, 0.1).y))

    def test_WavesDir(self):

        # Test the error on wrong input type
        with self.assertRaises(TypeError):
            gwdir = cw.GravitationalWavesDir(0)

        with self.assertRaises(TypeError):
            emdir = cw.ElectromagneticWavesDir(0)

        sim = sd.SimDir("tests/tov")
        gwdir = cw.GravitationalWavesDir(sim)
        emdir = cw.ElectromagneticWavesDir(sim)

        # Check type
        self.assertTrue(isinstance(gwdir[110.69],
                                   cw.GravitationalWavesOneDet))
        self.assertTrue(isinstance(emdir[110.69],
                                   cw.ElectromagneticWavesOneDet))
