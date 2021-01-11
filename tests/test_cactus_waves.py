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

import unittest

import numpy as np
from scipy import signal

from kuibit import cactus_waves as cw
from kuibit import gw_utils as gwu
from kuibit import simdir as sd
from kuibit import timeseries as ts


class TestCactusWaves(unittest.TestCase):
    def setUp(self):
        self.sim = sd.SimDir("tests/tov")
        self.gwdir = cw.GravitationalWavesDir(self.sim)
        self.psi4 = self.gwdir[110.69]

    def test_WavesOneDet(self):

        t1 = np.linspace(0, np.pi, 100)
        t2 = np.linspace(2 * np.pi, 3 * np.pi, 100)
        y1 = np.sin(t1)
        y2 = np.sin(t2)
        ts1 = ts.TimeSeries(t1, y1)
        ts2 = ts.TimeSeries(t2, y2)
        dist1 = 100

        data1 = [(2, 2, ts1), (2, 2, ts2)]
        data2 = [(1, 1, ts1), (1, 0, ts2), (1, -1, ts2)]

        gw = cw.GravitationalWavesOneDet(dist1, data1)
        em = cw.ElectromagneticWavesOneDet(dist1, data2)

        self.assertEqual(gw.l_min, 2)
        self.assertEqual(em.l_min, 1)

    def test_get_psi4_lm(self):

        self.assertEqual(
            self.gwdir[110.69].get_psi4_lm(2, 2), self.gwdir[110.69][(2, 2)]
        )

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

        # The integral of sin should be -cos
        # The period of sin(x) is 2 pi, so we pick pcut = 1e10
        integral = gwdum._fixed_frequency_integrated(tts, 1e10)

        self.assertTrue(np.allclose(integral.t, t))
        self.assertTrue(np.allclose(integral.y, -np.cos(t), atol=5e-4))

        # The second integral should be sin(x)
        integral2 = gwdum._fixed_frequency_integrated(tts, 1e10, order=2)

        self.assertTrue(np.allclose(integral2.y, -np.sin(t), atol=5e-4))

        # Now, let's see the opposite case in which the frequency is lower than
        # any frequencies. The output should be the same timeseries we started
        # with, rescaled by 1/(i omega_threshold). Now the factor of 1/i
        # corresponds to a rotation of pi/2, so we expect the output to be the
        # cosine. pcut = 1e-4 -> omega_threshold = 2 pi / pcut = 2 pi * 1e4
        # Hence, the timeseries is divided by 1e-4
        integral3 = gwdum._fixed_frequency_integrated(tts, 1e-4)
        self.assertTrue(
            np.allclose(integral3.y * 2 * np.pi * 1e4, -np.cos(t), atol=1e-3)
        )

        # Check warning for irregularly spaced
        with self.assertWarns(RuntimeWarning):
            tts.t[1] *= 1.01
            gwdum._fixed_frequency_integrated(tts, 1e-4)

    def test_get_strain_lm(self):

        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(3, 3, 1)

        # Large pcut
        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(2, 2, 1000)

        # Test window function errors
        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(2, 2, 0.1, window_function=1)
        # Not implemented
        with self.assertRaises(ValueError):
            self.psi4.get_strain_lm(2, 2, 0.1, window_function="bubu")

        # We do not need to test the FFI, hopefully that is already tested

        # Test window = set verything to 0
        self.assertTrue(
            np.allclose(
                self.psi4.get_strain_lm(
                    2, 2, 0.1, window_function=lambda x: 0
                ).y,
                0,
            )
        )

        psi4lm = self.psi4[(2, 2)]
        psi4lm *= self.psi4.dist

        # Test when window is a function
        ham_array = signal.hamming(len(psi4lm))
        ham_psi4lm = psi4lm.copy()
        ham_psi4lm.y *= ham_array
        ffi_ham = self.psi4._fixed_frequency_integrated(
            ham_psi4lm, 0.1, order=2
        ).y

        self.assertTrue(
            np.allclose(
                self.psi4.get_strain_lm(
                    2, 2, 0.1, window_function=signal.hamming, trim_ends=False
                ).y,
                ffi_ham,
            )
        )

        # Test window is a string, like hamming
        self.assertTrue(
            np.allclose(
                self.psi4.get_strain_lm(
                    2, 2, 0.1, window_function="hamming", trim_ends=False
                ).y,
                ffi_ham,
            )
        )

        # Test no window
        self.assertTrue(
            np.allclose(
                self.psi4.get_strain_lm(2, 2, 0.1, trim_ends=False).y,
                self.psi4._fixed_frequency_integrated(psi4lm, 0.1, order=2).y,
            )
        )

    def test_get_strain(self):

        # test l_max too big
        with self.assertRaises(ValueError):
            self.psi4.get_strain(0, 0, 1, l_max=100)

        theta, phi = np.pi / 2, 1
        ym2 = gwu.sYlm(-2, 2, -2, theta, phi)
        ym1 = gwu.sYlm(-2, 2, -1, theta, phi)
        y0 = gwu.sYlm(-2, 2, 0, theta, phi)
        y1 = gwu.sYlm(-2, 2, 1, theta, phi)
        y2 = gwu.sYlm(-2, 2, 2, theta, phi)

        strain = (
            self.psi4.get_strain_lm(2, -2, 0.1).y * ym2
            + self.psi4.get_strain_lm(2, -1, 0.1).y * ym1
            + self.psi4.get_strain_lm(2, 0, 0.1).y * y0
            + self.psi4.get_strain_lm(2, 1, 0.1).y * y1
            + self.psi4.get_strain_lm(2, 2, 0.1).y * y2
        )

        self.assertTrue(
            np.allclose(strain, self.psi4.get_strain(theta, phi, 0.1).y)
        )

    def test_get_observed_strain(self):

        # Let's check with Hanford
        theta_GW, phi_GW = np.pi / 3, 0
        antennas = gwu.antenna_responses_from_sky_localization(
            8, -70, "2015-09-14 09:50:45"
        )
        Fc_H, Fp_H = antennas.hanford

        expected_strain = self.psi4.get_strain(
            theta_GW, phi_GW, 0.1, trim_ends=False
        )
        expected_strain = (
            expected_strain.real() * Fp_H - expected_strain.imag() * Fc_H
        )

        strain = self.psi4.get_observed_strain(
            8,
            -70,
            "2015-09-14 09:50:45",
            theta_GW,
            phi_GW,
            0.1,
            trim_ends=False,
        )

        self.assertEqual(strain.hanford, expected_strain)

    def test_get_power_energy(self):

        psi4lm = self.psi4[(2, 2)]

        psi4lm_int = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=1
        )

        power_lm = self.psi4.dist ** 2 / (16 * np.pi) * np.abs(psi4lm_int) ** 2

        self.assertEqual(power_lm, self.psi4.get_power_lm(2, 2, 0.1))
        self.assertEqual(
            power_lm.integrated(), self.psi4.get_energy_lm(2, 2, 0.1)
        )

        # Total power
        total_power = (
            self.psi4.get_power_lm(2, 2, 0.1)
            + self.psi4.get_power_lm(2, 1, 0.1)
            + self.psi4.get_power_lm(2, 0, 0.1)
            + self.psi4.get_power_lm(2, -1, 0.1)
            + self.psi4.get_power_lm(2, -2, 0.1)
        )

        self.assertEqual(total_power, self.psi4.get_total_power(0.1))
        self.assertEqual(
            total_power.integrated(), self.psi4.get_total_energy(0.1)
        )

    def test_get_power_energy_em(self):

        emdir = cw.ElectromagneticWavesDir(self.sim)
        phi2 = emdir[110.69]
        phi2lm = phi2[(2, 2)]

        power_lm = phi2.dist ** 2 / (4 * np.pi) * np.abs(phi2lm) ** 2

        self.assertEqual(power_lm, phi2.get_power_lm(2, 2))
        self.assertEqual(power_lm.integrated(), phi2.get_energy_lm(2, 2))

        # Total power
        total_power = (
            phi2.get_power_lm(2, 2)
            + phi2.get_power_lm(2, 1)
            + phi2.get_power_lm(2, 0)
            + phi2.get_power_lm(2, -1)
            + phi2.get_power_lm(2, -2)
        )

        self.assertEqual(total_power, phi2.get_total_power())
        self.assertEqual(total_power.integrated(), phi2.get_total_energy())

    def test_get_torque_angular_momentum(self):

        psi4lm = self.psi4[(2, 2)]

        psi4lm_int1 = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=1
        )
        psi4lm_int2 = self.psi4._fixed_frequency_integrated(
            psi4lm, 0.1, order=2
        )

        torque_lm = (
            self.psi4.dist ** 2
            / (16 * np.pi)
            * 2
            * (np.conj(psi4lm_int2) * psi4lm_int1).imag()
        )

        self.assertEqual(torque_lm, self.psi4.get_torque_z_lm(2, 2, 0.1))
        self.assertEqual(
            torque_lm.integrated(),
            self.psi4.get_angular_momentum_z_lm(2, 2, 0.1),
        )

        # Total power
        total_torque_z = (
            self.psi4.get_torque_z_lm(2, 2, 0.1)
            + self.psi4.get_torque_z_lm(2, 1, 0.1)
            + self.psi4.get_torque_z_lm(2, 0, 0.1)
            + self.psi4.get_torque_z_lm(2, -1, 0.1)
            + self.psi4.get_torque_z_lm(2, -2, 0.1)
        )

        self.assertEqual(total_torque_z, self.psi4.get_total_torque_z(0.1))
        self.assertEqual(
            total_torque_z.integrated(),
            self.psi4.get_total_angular_momentum_z(0.1),
        )

    def test_WavesDir(self):

        # Test the error on wrong input type
        with self.assertRaises(TypeError):
            cw.GravitationalWavesDir(0)

        with self.assertRaises(TypeError):
            cw.ElectromagneticWavesDir(0)

        emdir = cw.ElectromagneticWavesDir(self.sim)

        # Check type
        self.assertTrue(
            isinstance(self.gwdir[110.69], cw.GravitationalWavesOneDet)
        )
        self.assertTrue(
            isinstance(emdir[110.69], cw.ElectromagneticWavesOneDet)
        )

    def test_extrapolation(self):

        # Test too many radii for the extrapolation
        with self.assertRaises(RuntimeError):
            self.gwdir._extrapolate_waves_to_infinity(
                [1, 2], [1, 2], [1, 2], 1, order=3
            )

        # Number of radii is not the same as waves
        with self.assertRaises(RuntimeError):
            self.gwdir._extrapolate_waves_to_infinity(
                [1, 2], [1, 2], [1, 2, 3], 1, order=1
            )

        # TODO: Write real tests for these functions. Compare with POWER?

        # NOTE: These tests are not physical tests!

        # Smoke test, we just check that the code returns expected data types
        times = np.linspace(0, 2 * np.pi, 100)
        sins = ts.TimeSeries(times, np.sin(times))
        sins_plus_one = ts.TimeSeries(times, np.sin(times + 1))
        self.gwdir._extrapolate_waves_to_infinity(
            [sins, sins_plus_one],
            np.linspace(-12, -11, 100),
            [10, 11],
            1,
            order=1,
        )

        self.gwdir.extrapolate_strain_lm_to_infinity(
            2, 2, 0.1, [110.69, 110.69], [2791, 2791.1], order=0
        )
        self.gwdir.extrapolate_strain_lm_to_infinity(
            2,
            2,
            0.1,
            [110.69, 110.69],
            [2791, 2791.1],
            order=0,
            extrapolate_amplitude_phase=True,
        )

    def test_empty(self):

        # Test with no wave information
        empty_sim = sd.SimDir("kuibit")
        self.assertIs(empty_sim.gws.l_max, None)
